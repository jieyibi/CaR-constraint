import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import copy
import re
import csv
import pytz
import pdb
import wandb
import itertools
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from tensorboard_logger import Logger as TbLogger
from sklearn.metrics import confusion_matrix
from utils import *
from models.SINGLEModel import SINGLEModel, Critic_Construct, Critic_N2S
# from sklearn.utils.class_weight import compute_class_weight
# from torch.utils.data import DataLoader, DistributedSampler  # use pytorch dataloader
# from sklearn.metrics import confusion_matrix, roc_auc_score


class Trainer:
    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params, tester_params, rank):
        # save arguments
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.tester_params = tester_params
        self.problem = self.args.problem

        self.device = torch.device(f'cuda:{rank}')
        self.rank = rank
        self.world_size = self.args.world_size
        self.trainer_params['train_episodes'] = self.trainer_params['train_episodes'] // self.world_size
        self.log_path = args.log_path
        self.result_log = {"val_score": [], "val_gap": [], "val_infsb_rate": []}
        if self.trainer_params["validation_improve_steps"] > 0.:
            self.result_log.update({"val_score_improve": [], "val_gap_improve": [], "val_infsb_rate_improve": []})
            if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]:
                self.result_log.update({"rc_val_score": [], "rc_val_gap": [], "rc_val_infsb_rate": []})
        if self.tester_params["aux_mask"]:
            self.result_log.update({"rc_masked_val_score": [], "rc_masked_val_gap": [], "rc_masked_val_infsb_rate": []})
        if args.tb_logger and rank == 0:
            self.tb_logger = TbLogger(self.log_path)
        else:
            self.tb_logger = None
        self.best_score_cons = {}
        self.best_score_impr = {}

        # Main Components
        self.envs = get_env(self.args.problem)
        self.model = get_model(self.args.model_type)(**self.model_params).to(self.device)
        if self.args.multiple_gpu:
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True, static_graph=True)
        if self.args.uncertainty_weight:
            self.loss_fn = MultiTaskLoss()
            self.optimizer = Optimizer(list(self.model.parameters()) + list(self.loss_fn.parameters()),
                                       **self.optimizer_params['optimizer'])
        elif trainer_params["reward_gating"]:
            self.lambda_ = nn.Parameter(torch.ones((trainer_params["constraint_number"],), requires_grad=True))
            self.optimizer = Optimizer([{'params': self.model.parameters()}, {'params': [self.lambda_]}],
                                       **self.optimizer_params['optimizer'])
        else:
            # [torch.zeros([self.args.train_batch_size, 40])]
            self.lambda_ = [torch.tensor(0.)] * trainer_params["constraint_number"] if trainer_params["subgradient"] else self.trainer_params["penalty_factor"]
            self.subgradient_lr = trainer_params["subgradient_lr"]
            if self.trainer_params["shared_critic"]:
                self.critic_construct = Critic_Construct()
                self.critic_improve = Critic_N2S(
                    embedding_dim=128,
                    ff_hidden_dim=128,
                    n_heads=4,
                    n_layers=3,
                    normalization="layer",
                )  # follow NCS
                if self.args.multiple_gpu:
                    self.critic_construct = DDP(self.critic_construct, device_ids=[rank], find_unused_parameters=True, static_graph=True)
                    self.critic_improve = DDP(self.critic_improve, device_ids=[rank], find_unused_parameters=True, static_graph=True)
                self.optimizer = Optimizer([{'params': self.model.parameters(), 'lr': args.lr}] +
                                           [{'params': self.critic_construct.parameters(), 'lr': args.lr_critic_cons}] +
                                           [{'params': self.critic_improve.parameters(), 'lr': args.lr_critic_impr}]
                                           , **self.optimizer_params['optimizer'])
                print(f'>> use shared critic: [CONS] {args.lr_critic_cons}; [IMPR] {args.lr_critic_impr}!')
                num_param(self.critic_construct)
                num_param(self.critic_improve)
            else:
                self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.scaler = torch.cuda.amp.GradScaler()
        num_param(self.model)

        if self.model_params["pip_decoder"]:
            self.is_train_pip_decoder = True if self.trainer_params["simulation_stop_epoch"] > 0 else False
            self.accuracy_bsf, self.fsb_accuracy_bsf, self.infsb_accuracy_bsf =  0., 0., 0.
            self.accuracy_isbsf, self.fsb_accuracy_isbsf, self.infsb_accuracy_isbsf = False, False, False

            self.train_sl_epoch_list = list(range(1, self.trainer_params["simulation_stop_epoch"] + 1))

            for start in range(self.trainer_params["pip_update_interval"], self.trainer_params["epochs"] + 1, self.trainer_params["pip_update_interval"]):
                self.train_sl_epoch_list.extend(range(start - self.trainer_params["pip_update_epoch"] + 1, start + 1))

            if self.trainer_params["pip_last_growup"] > self.trainer_params["pip_update_epoch"]:
                self.train_sl_epoch_list.extend(range(self.trainer_params["epochs"] - self.trainer_params["pip_last_growup"] + 1, self.trainer_params["epochs"]+1))

            self.load_sl_epoch_list = [self.trainer_params["simulation_stop_epoch"] + 1] + list(range(1, self.trainer_params["epochs"] - self.trainer_params["pip_last_growup"] + 1, self.trainer_params["pip_update_interval"]))[1:]

            # print(self.train_sl_epoch_list)
            # print(self.load_sl_epoch_list)

            # PIP decoder does not update frequently,
            # Hence we record the latest updated one and use it to predict PI mask until the next update
            if self.trainer_params["lazy_pip_model"]:
                self.lazy_model = SINGLEModel(**self.model_params)
            else:
                self.lazy_model = None

            if args.pip_checkpoint:
                checkpoint_fullname = args.pip_checkpoint
                checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
                try:
                    self.lazy_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                except:
                    self.lazy_model.load_state_dict(checkpoint, strict=True)
                try:
                    print(
                        ">> Load lazy PIP-D model from {} [Accuracy: {:.4f}%; Infeasible: {:.4f}%; Feasible: {:.4f}%]".format(
                            checkpoint_fullname, checkpoint['accuracy'] * 100, checkpoint['infsb_accuracy'] * 100,
                            checkpoint['fsb_accuracy'] * 100))
                    if "fsb_accuracy_bsf.pt" in checkpoint_fullname:
                        self.fsb_accuracy_bsf = checkpoint['fsb_accuracy']
                    elif "infsb_accuracy_bsf.pt" in checkpoint_fullname:
                        self.infsb_accuracy_bsf = checkpoint['infsb_accuracy']
                    else:
                        self.accuracy_bsf = checkpoint['accuracy']
                except:
                    print(">> Load lazy PIP-D model from {}".format(checkpoint_fullname))

        self.penalty_factor = self.trainer_params["penalty_factor"]

        # Restore
        self.start_epoch = 1
        if args.checkpoint is not None:
            checkpoint_fullname = args.checkpoint
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            # ckpt = {'model_state_dict': checkpoint['model_state_dict']}
            # torch.save(ckpt, "checkpoint_fullname"+".ckpt")
            if 'model_state_dict' in checkpoint.keys():
                checkpoint_tmp = checkpoint['model_state_dict']
            else:
                checkpoint_tmp = checkpoint
            try:
                self.model.load_state_dict(checkpoint_tmp, strict=True)
                if self.trainer_params["shared_critic"]:
                    self.critic_construct.load_state_dict(checkpoint["critic_construct"], strict=True)
                    self.critic_improve.load_state_dict(checkpoint["critic_improve"], strict=True)
            except:
                # TODO: add critic
                try:
                    # from single-gpu to multi-gpus
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint_tmp.items():
                        name = 'module.' + k # add `module.`
                        new_state_dict[name] = v
                    self.model.load_state_dict({**new_state_dict}, strict=True)
                except:
                    # from multi-gpus to single-gpu
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint_tmp.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    self.model.load_state_dict({**new_state_dict}, strict=True)
            self.start_epoch = 1 + checkpoint['epoch']
            self.scheduler.last_epoch = checkpoint['epoch'] - 1
            if self.trainer_params["load_optimizer"]:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Rank {rank} >> optimizer (Epoch: {checkpoint['epoch']}) Loaded (lr = {self.optimizer.param_groups[0]['lr']})!")
            print(f"Rank {rank} >> Checkpoint (Epoch: {checkpoint['epoch']}) Loaded!")
            print(f"Rank {rank} >> Load from {checkpoint_fullname}")

        if args.POMO_checkpoint is not None and args.init_sol_strategy == "POMO" and args.improvement_only:
            input_model_params = {**self.model_params, "improve_steps": 0, "improvement_only": False} # only construction
            self.pomo_model = get_model(self.args.model_type)(**input_model_params).to(self.device)
            print(f"Rank {rank} >> Load from {args.POMO_checkpoint}")
            num_param(self.pomo_model)
            checkpoint = torch.load(args.POMO_checkpoint, map_location=self.device)
            if 'model_state_dict' in checkpoint.keys(): checkpoint = checkpoint['model_state_dict']
            try:
                self.pomo_model.load_state_dict(checkpoint, strict=True)
            except:
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.pomo_model.load_state_dict({**new_state_dict}, strict=True)
            self.pomo_model.eval()

        # utility
        self.time_estimator = TimeEstimator()

        self.binary_string_pool = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=model_params['z_dim'])])

    def train(self):
        self.time_estimator.reset(self.start_epoch)
        self.improve_steps = copy.deepcopy(self.trainer_params["improve_steps"])
        self.validation_improve_steps = copy.deepcopy(self.trainer_params["validation_improve_steps"])
        if self.trainer_params["improve_start_when_dummy_ok"] and self.trainer_params["improve_steps"] > 0: print(">> Improvement will begin after the number of depots in the constructed routes is less than {}".format(self.trainer_params["max_dummy_size"]))
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            if self.rank==0: print('================================================================================')

            # Load the latest updated PIP model for PI masking prediction
            if self.model_params["pip_decoder"]:
                if epoch not in self.train_sl_epoch_list:
                    if self.rank == 0: print('>> PIP decoder is not training...')
                    self.is_train_pip_decoder = False
                    self.model_params["generate_PI_mask"] = False
                    if self.trainer_params["lazy_pip_model"] and (epoch in self.load_sl_epoch_list) and epoch != self.start_epoch: # if epoch == start_epoch, ckpt already loaded or it is training (no need to load)
                        pip_checkpoint = {"last_epoch": "epoch-{}.pt".format(epoch - 1),
                                           "train_fsb_bsf": "fsb_accuracy_bsf.pt",
                                           "train_infsb_bsf": "infsb_accuracy_bsf.pt",
                                           "train_accuracy_bsf": "accuracy_bsf.pt"}
                        checkpoint_fullname = os.path.join(self.log_path, pip_checkpoint[self.trainer_params["load_which_pip"]])
                        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
                        try:
                            self.lazy_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                        except:
                            self.lazy_model.load_state_dict(checkpoint, strict=True)
                        try:
                            if self.rank == 0: print(">> Load lazy PIP-D model from {} [Accuracy: {:.4f}%; Infeasible: {:.4f}%; Feasible: {:.4f}%]".format(checkpoint_fullname, checkpoint['accuracy']*100, checkpoint['infsb_accuracy']*100, checkpoint['fsb_accuracy']*100))
                        except:
                            if self.rank == 0: print(">> Load lazy PIP-D model from {}".format(checkpoint_fullname))
                else:
                    if self.rank == 0: print('>> PIP decoder is training...')
                    self.is_train_pip_decoder = True
                    self.model_params["generate_PI_mask"] = True

            if self.trainer_params["non_linear"] == "decayed_epsilon":
                self.trainer_params["epsilon"] = self.trainer_params["epsilon_base"] * np.exp(- self.trainer_params["epsilon_decay_beta"] * epoch)
                print(f'>> Epsilon = {self.trainer_params["epsilon"]}')
            # if epoch == 401:
            #     self.trainer_params["select_top_k"] = 3
            # else:
            #     self.trainer_params["select_top_k"] = 4

            # if dummy <= max_dummy_size, begin to improve
            # print("{}, {}".format(improve_steps, validation_improve_steps))
            if self.trainer_params["improve_start_when_dummy_ok"]:
                if self.start_epoch == 1 and (epoch == 1 or (dummy_size > self.trainer_params["max_dummy_size"])):
                    self.trainer_params["improve_steps"] = 0
                    self.trainer_params["validation_improve_steps"] = 0
                else:
                    self.trainer_params["improve_steps"] = self.improve_steps
                    self.trainer_params["validation_improve_steps"] = self.validation_improve_steps
            # print(self.trainer_params["improve_steps"], self.trainer_params["validation_improve_steps"])

            # initialize logger
            self.metric_logger = metric_logger(self.problem, self.model_params["pip_decoder"])

            # Train
            self._train_one_epoch(epoch)
            dummy_size = self.metric_logger.dummy_size.avg
            print("dummy_size: {}".format(dummy_size))

            # Step
            self.scheduler.step()

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            if self.rank==0: print(f"Rank {self.rank} >> Epoch {epoch}/{self.trainer_params['epochs']}: Time Est.: Elapsed[{elapsed_time_str}], Remain[{remain_time_str}]")
            if self.args.wandb_logger and self.rank == 0: self._log_in_wandb(epoch)
            if self.tb_logger and self.rank == 0: self._log_in_tb_logger(epoch)

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['model_save_interval']
            validation_interval = self.trainer_params['validation_interval']

            # MTL Validation & save latest images
            self._save_best_model(score_cons = self.metric_logger.construct_metrics["score"].avg, score_impr = self.metric_logger.improve_metrics["current_score"].avg, mode="train")
            if all_done or (epoch % model_save_interval == 0):
                if self.rank == 0:
                    print(f"Rank {self.rank} >> Saving trained_model")
                    checkpoint_dict = {
                        'epoch': epoch,
                        'problem': self.args.problem,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log,
                        'metrics_logger': self.metric_logger,
                        'critic_construct': self.critic_construct.state_dict() if self.trainer_params["shared_critic"] else {},
                        'critic_improve': self.critic_improve.state_dict() if self.trainer_params["shared_critic"] else {},
                    }
                    torch.save(checkpoint_dict, '{}/epoch-{}.pt'.format(self.log_path, epoch))

            # validation
            if epoch == 1 or (epoch % validation_interval == 0):
            # if (epoch % validation_interval == 0):

                # Load validation dataset
                val_problems = [self.args.problem]
                val_episodes, problem_size = self.env_params['val_episodes'], self.env_params['problem_size']
                if self.env_params['val_dataset'] is not None:
                    paths = self.env_params['val_dataset']
                    dir = ["./data/{}/".format(self.args.problem)] * len(paths)
                    val_envs = [get_env(prob)[0] for prob in val_problems] * len(paths)
                else:
                    dir = [os.path.join("./data", prob) for prob in val_problems]
                    paths = ["{}{}_uniform.pkl".format(prob.lower(), problem_size) for prob in val_problems]
                    val_envs = [get_env(prob)[0] for prob in val_problems]

                # Validate
                if self.trainer_params["validation_improve_steps"] > 0 and not self.env_params["pomo_start"] and self.trainer_params["val_pomo_size"]>10:
                    if (epoch % 1000 == 0):
                        val_episodes = self.env_params['val_episodes'] = 10000
                    else:
                        val_episodes = self.env_params['val_episodes'] = 1000
                print(">> Val {} instances.".format(val_episodes))
                for i, path in enumerate(paths):
                    # if no optimal solution provided, set compute_gap to False
                    if not self.env_params["pomo_start"] and self.trainer_params["train_z_sample_size"] == 0: # sample X solutions for each instance
                        # sampling pomo_size routes is useless due to the argmax operator when selecting next node based on probability
                        init_pomo_size = self.env_params["pomo_size"]
                        self.env_params["pomo_size"] = self.trainer_params["val_pomo_size"]

                    self._val_and_stat(dir[i], path, val_envs[i](**self.env_params), batch_size=self.trainer_params["validation_batch_size"], val_episodes=val_episodes, epoch = epoch)
                    # score, gap = self._val_and_stat(dir[i], path, val_envs[i](**{"problem_size": problem_size, "pomo_size": problem_size}), batch_size=500, val_episodes=val_episodes, compute_gap=True)

                    if not self.env_params["pomo_start"] and self.trainer_params["train_z_sample_size"] == 0:
                        self.env_params["pomo_size"] = init_pomo_size

                    # log
                    self._val_logger(epoch)

                    # save
                    self._save_best_model(score_cons=self.val_metric_logger.construct_metrics["aug_gap_list"], score_impr=self.val_metric_logger.improve_metrics["aug_gap_list"], mode="val")

    def test(self):

        # Load test dataset
        test_problems = [self.args.problem]
        test_episodes, problem_size = self.tester_params['test_episodes'], self.env_params['problem_size']
        if self.tester_params['test_dataset'] is not None:
            paths = self.tester_params['test_dataset']
            dir = ["./data/{}/".format(self.args.problem)] * len(paths)
            test_envs = [get_env(prob)[0] for prob in test_problems] * len(paths)
        else:
            dir = [os.path.join("./data", prob) for prob in test_problems]
            paths = ["{}{}_uniform.pkl".format(prob.lower(), problem_size) for prob in test_problems]
            test_envs = [get_env(prob)[0] for prob in test_problems]


        results_out=[]
        for i, path in enumerate(paths):
            print(">> TEST dataset {}".format(path))
            print(">> TEST {} instances.".format(test_episodes))
            start_time = time.time()
            # if no optimal solution provided, set compute_gap to False
            if not self.env_params["pomo_start"] and self.tester_params["test_z_sample_size"] == 0: # can only get one solution for each instance due to greedy strategy
                # sampling pomo_size routes is useless due to the argmax operator when selecting next node based on probability
                init_pomo_size = self.env_params["pomo_size"]
                self.env_params["pomo_size"] = self.tester_params["test_pomo_size"] #for improvement

            self._val_and_stat(dir[i], path, test_envs[i](**self.env_params), batch_size=self.tester_params["test_batch_size"],
                               val_episodes=test_episodes, epoch=1)
            print(">> Evaluation finished within {:.2f}s\n".format(time.time() - start_time))

            if self.trainer_params["improve_steps"] > 0:
                score = self.val_metric_logger.improve_metrics["aug_score_list"] * 1000
            else:
                score = self.val_metric_logger.construct_metrics["aug_score_list"] * 1000
            results_out.append([path.split("/")[-1].split(".")[0], score])

            print(" \n*** Test Done on {} *** ".format(self.args.problem))

            print(" \n*** Construction *** ")
            print(" NO-AUG SCORE: {}, Gap: {:.4f} ".format(self.val_metric_logger.construct_metrics["no_aug_score_list"],
                                                           self.val_metric_logger.construct_metrics["no_aug_gap_list"]))
            print(" AUGMENTATION SCORE: {}, Gap: {:.4f} ".format(self.val_metric_logger.construct_metrics["aug_score_list"],
                                                                 self.val_metric_logger.construct_metrics["aug_gap_list"]))
            print("Solution level Infeasible rate: {:.3f}%".format(self.val_metric_logger.construct_metrics["sol_infeasible_rate_list"]))
            print("Instance level Infeasible rate: {:.3f}%".format(self.val_metric_logger.construct_metrics["ins_infeasible_rate_list"]))

            if self.trainer_params["improve_steps"] >0:
                print(" \n*** Improvement *** ")
                print(" NO-AUG SCORE: {}, Gap: {:.4f} ".format(self.val_metric_logger.improve_metrics["no_aug_score_list"],
                                                               self.val_metric_logger.improve_metrics["no_aug_gap_list"]))
                print(" AUGMENTATION SCORE: {}, Gap: {:.4f} ".format(self.val_metric_logger.improve_metrics["aug_score_list"],
                                                                     self.val_metric_logger.improve_metrics["aug_gap_list"]))
                print("Solution level Infeasible rate: {:.3f}%".format(self.val_metric_logger.improve_metrics["sol_infeasible_rate_list"]))
                print("Instance level Infeasible rate: {:.3f}%".format(self.val_metric_logger.improve_metrics["ins_infeasible_rate_list"]))

            if self.problem == "VRPBLTW":
                print(" \n*** Re-Construction w. mask *** ")
                print(" NO-AUG SCORE: {}, Gap: {:.4f} ".format(self.val_metric_logger.reconstruct_masked_metrics["no_aug_score_list"],
                                                               self.val_metric_logger.reconstruct_masked_metrics["no_aug_gap_list"]))
                print(" AUGMENTATION SCORE: {}, Gap: {:.4f} ".format(self.val_metric_logger.reconstruct_masked_metrics["aug_score_list"],
                                                                     self.val_metric_logger.reconstruct_masked_metrics["aug_gap_list"]))
                print("Solution level Infeasible rate: {:.3f}%".format(self.val_metric_logger.reconstruct_masked_metrics["sol_infeasible_rate_list"]))
                print("Instance level Infeasible rate: {:.3f}%".format(self.val_metric_logger.reconstruct_masked_metrics["ins_infeasible_rate_list"]))

            if not self.env_params["pomo_start"] and self.tester_params["test_z_sample_size"] == 0:
                self.env_params["pomo_size"] = init_pomo_size

            # log
            self._val_logger(epoch=1)

            if self.tester_params["is_lib"]:
                df = pd.DataFrame(np.array(results_out))
                excel_file = "cvrplib_car100_new_train_eas200_10_softmax16.xlsx"
                df.to_excel(excel_file, index=False, header=False)
                print(df)

    def _train_one_epoch(self, epoch):
        episode = 0

        train_num_episode = self.trainer_params['train_episodes']
        total_step = math.floor(train_num_episode /self.trainer_params['train_batch_size'])
        batch_id = 0
        batch_reward = None
        weights = 0
        # reset the metrics of PIP-D performance every epoch
        if self.model_params["pip_decoder"] and self.is_train_pip_decoder: self.metric_logger._reset_pip_d_metrics()

        while episode < train_num_episode:
            # if self.rank==0: print(episode)
            self.episode = episode
            self.epoch = epoch
            for accumulation_step in range(self.trainer_params['accumulation_steps']):
                remaining = train_num_episode - episode
                batch_size = min(self.trainer_params['train_batch_size'], remaining)

                env = random.sample(self.envs, 1)[0](**self.env_params)
                data = env.get_random_problems(batch_size, self.env_params["problem_size"])

                if self.trainer_params["improve_steps"] > 0.:
                    if batch_reward is None:
                        batch_reward = []
                        weights = 0
                    else:
                        try:
                            batch_reward = torch.cat(batch_reward)
                            if self.args.multiple_gpu:
                                dist.barrier()
                                batch_reward = gather_tensor_and_concat(batch_reward.contiguous())
                                dist.barrier()
                            weights = batch_reward.mean()
                            batch_reward = []
                        except:
                            batch_reward = []
                            weights = 0

                sl_output = self._train_one_batch(data, env, batch_reward, weights, accumulation_step=accumulation_step)

                if sl_output is not None and self.model_params["pip_decoder"]:
                    sl_loss, accuracy, infsb_accuracy, infsb_samples, fsb_accuracy, fsb_samples = sl_output

                    self.metric_logger.construct_metrics["sl_loss"].update(sl_loss, infsb_samples+fsb_samples)
                    self.metric_logger.construct_metrics["accuracy"].update(accuracy, infsb_samples+fsb_samples)
                    self.metric_logger.construct_metrics["infsb_accuracy"].update(infsb_accuracy, infsb_samples)
                    self.metric_logger.construct_metrics["fsb_accuracy"].update(fsb_accuracy, fsb_samples)

                    # never log in batches [Different from Bi et. al, 2024.]
                    # if self.args.tb_logger:
                    #     self.tb_logger.log_value('sl_batch/sl_loss', sl_loss, (epoch-1) * total_step + batch_id)
                    #     self.tb_logger.log_value('sl_batch/accuracy', accuracy, (epoch-1) * total_step + batch_id)
                    #     self.tb_logger.log_value('sl_batch/infsb_accuracy', infsb_accuracy, (epoch-1) * total_step + batch_id)
                    #     self.tb_logger.log_value('sl_batch/infsb_samples_number', infsb_samples, (epoch-1) * total_step + batch_id)
                    #     self.tb_logger.log_value('sl_batch/fsb_accuracy', fsb_accuracy, (epoch-1) * total_step + batch_id)
                    #     self.tb_logger.log_value('sl_batch/fsb_samples_number', fsb_samples, (epoch-1) * total_step + batch_id)
                    # if self.args.wandb_logger:
                    #     wandb.log({'sl_batch/sl_loss': sl_loss})
                    #     wandb.log({'sl_batch/accuracy': accuracy})
                    #     wandb.log({'sl_batch/infsb_accuracy': infsb_accuracy})
                    #     wandb.log({'sl_batch/infsb_samples_number': infsb_samples})
                    #     wandb.log({'sl_batch/fsb_accuracy': fsb_accuracy})
                    #     wandb.log({'sl_batch/fsb_samples_number': fsb_samples})

                # torch.cuda.empty_cache()

                episode += batch_size
                batch_id += 1

                if episode >= train_num_episode:
                    break

        # Log Once, for each epoch
        self._print_log()
        if self.model_params["pip_decoder"] and self.is_train_pip_decoder and self.rank == 0: self._save_pip_decoder(accuracy, infsb_accuracy, fsb_accuracy)

    def _train_one_batch(self, data, env, batch_reward=None, weights=0, accumulation_step=1, is_test_sl=False):
        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        amp_training = self.trainer_params['amp_training']
        if self.model_params["polynet"]:
            z_sample_size = self.trainer_params['train_z_sample_size']
            z_dim = self.model_params['z_dim']
            if self.env_params['pomo_start']:
                starting_points = self.env_params['problem_size']
                rollout_size = starting_points * z_sample_size
            else:
                starting_points = 1
                rollout_size = z_sample_size
            # Sample z vectors
            z = self.sample_z_vectors(batch_size, starting_points, z_dim, z_sample_size, rollout_size)
            # shape: (batch_size, rollout_size, z_dim)
        else:
            rollout_size = self.env_params["pomo_size"]
            z = None

        self.model.train()
        try:
            self.model.module.set_eval_type(self.model_params["eval_type"])
        except:
            self.model.set_eval_type(self.model_params["eval_type"])

        if not (self.trainer_params["improvement_only"] and self.trainer_params["init_sol_strategy"]=="POMO"):
            env.load_problems(batch_size, rollout_size=rollout_size, problems=data, aug_factor=1)
            reset_state, _, _ = env.reset()

            # POMO Rollout
            state, reward, done = env.pre_step()
            # print("{}\n".format(state.PROBLEM))

        ###########################################Construction########################################
        ###########################################Construction########################################
        ###########################################Construction########################################
        if not self.model_params["improvement_only"]:
            try:
                self.model.module.pre_forward(reset_state, z)
                if self.model_params["pip_decoder"] and self.lazy_model is not None and (not self.is_train_pip_decoder):
                    self.lazy_model.eval()
                    self.lazy_model.module.pre_forward(reset_state)
            except:
                self.model.pre_forward(reset_state, z)
                if self.model_params["pip_decoder"] and self.lazy_model is not None and (not self.is_train_pip_decoder):
                    self.lazy_model.eval()
                    self.lazy_model.pre_forward(reset_state)

            # Initialize the prob list
            if self.model_params["dual_decoder"]:
                prob_list1 = torch.zeros(size=(batch_size, env.pomo_size, 0)).to(self.device)
                prob_list2 = torch.zeros(size=(batch_size, env.pomo_size, 0)).to(self.device)
            else:
                prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0)).to(self.device)
                probs_return_list = torch.zeros(size=(batch_size, env.pomo_size, env.problem_size+1, 0)).to(self.device) if self.trainer_params["probs_return"] else None
            if self.model_params["pip_decoder"] and self.is_train_pip_decoder:
                sl_loss_list = torch.zeros(size=(0,))
                pred_LIST, label_LIST = np.array([]), np.array([])
            # Start construction
            # tik = time.time()
            while not done:
                with torch.amp.autocast(device_type="cuda", enabled=amp_training):
                    selected, prob, probs_return = self.model(state, pomo=self.env_params["pomo_start"],
                                                              candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None,
                                                              return_probs=self.trainer_params["probs_return"])
                                                              # return_probs=self.trainer_params["reward_gating"])
                if probs_return is not None:
                    # constraint_mask = env.simulated_ninf_flag.to(self.device)
                    # mask_list = torch.cat((mask_list, constraint_mask[:, :, :, None]), dim=3)
                    probs_return_list = torch.cat((probs_return_list, probs_return[:, :, :, None]), dim=3)

                if self.model_params["dual_decoder"]:
                    prob1, prob2 = prob # shape: (batch, pomo)

                # Use PIP decoder to predict PI masking when the decoder is not trained.
                use_predicted_PI_mask = True if (self.model_params['pip_decoder'] and not self.is_train_pip_decoder) else False
                if self.model_params["pip_decoder"] and self.lazy_model is not None and env.selected_count >= 1 and (not self.is_train_pip_decoder):
                    with torch.no_grad():
                        use_predicted_PI_mask = self.lazy_model(state, candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None, use_predicted_PI_mask=False, no_select_prob=True)
                # Calculate the loss for the PIP decoder
                if self.model_params['pip_decoder']:
                    prob, probs_sl = prob
                    if self.model_params['pip_decoder'] and env.selected_count >= 1 and (env.selected_count < env.problem_size - 1) and self.is_train_pip_decoder:
                        visited_mask = env.visited_ninf_flag == float('-inf')
                        sl_losses = torch.tensor(0.)
                        label = torch.where(env.simulated_ninf_flag == float('-inf'), 1., env.simulated_ninf_flag)
                        label = label[~visited_mask]
                        if label.sum() != 0 and label.sum() != label.reshape(-1).size(-1):  # not all fsb or all infsb
                            probs_sl = probs_sl[~visited_mask]
                            infsb_sample_number = torch.nonzero(label != 0).size(0)  # positive
                            fsb_sample_number = torch.nonzero(label == 0).size(0)  # negative
                            pos_weight = fsb_sample_number / infsb_sample_number  # neg / pos
                            pos_weight = torch.ones_like(label) * pos_weight
                            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                            sl_loss = criterion(probs_sl, label)
                            sl_weight = (fsb_sample_number + infsb_sample_number) / (2 * fsb_sample_number)
                            # with this weight, fast method totally equals to the non-fast one
                            sl_loss = sl_loss * sl_weight
                            # sl_loss shape: (batch, pomo)
                            label = label.reshape(-1)
                            probs_sl = F.sigmoid(probs_sl).reshape(-1)
                            pred_LIST = np.append(pred_LIST, probs_sl.detach().cpu().numpy())
                            label_LIST = np.append(label_LIST, label.detach().cpu().numpy())
                            sl_losses += sl_loss
                        sl_loss_list = torch.cat([sl_loss_list, sl_losses.unsqueeze(0)], dim=0)
                # if True, then don't use predicted PI mask
                use_predicted_PI_mask = ((not isinstance(use_predicted_PI_mask, bool)  # if True, PI mask is predicted from the PIP decoder
                                          or use_predicted_PI_mask)  # PIP decoder isn't training
                                         or not self.trainer_params["use_real_PI_mask"])  # don't use real PI mask

                state, reward, done, infeasible = env.step(selected.to(self.device),
                                                           out_reward=self.trainer_params["out_reward"],
                                                           soft_constrained=self.trainer_params["soft_constrained"],
                                                           backhaul_mask=self.trainer_params["backhaul_mask"],
                                                           penalty_normalize=self.trainer_params["penalty_normalize"],
                                                           generate_PI_mask=self.trainer_params["generate_PI_mask"],
                                                           use_predicted_PI_mask=use_predicted_PI_mask,
                                                           pip_step=self.trainer_params["pip_step"]
                                                           )

                if self.model_params["dual_decoder"]:
                    prob_list1 = torch.cat((prob_list1, prob1[:, :, None]), dim=2)
                    prob_list2 = torch.cat((prob_list2, prob2[:, :, None]), dim=2)
                else:
                    prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2) # shape: (batch, pomo, solution)

            # Get construction output
            if self.model_params["dual_decoder"]:
                prob_list = (prob_list1, prob_list2)

        # if self.rank==0: print(f"Rank {self.rank} >> construction time: ", time.time() - tik)
        ###########################################Improvement########################################
        ###########################################Improvement########################################
        ###########################################Improvement########################################
        # tik = time.time()
        if "TSP" not in self.args.problem and not self.model_params["improvement_only"]: self.metric_logger.dummy_size.update(env.dummy_size, batch_size)
        start_sign = True
        if self.trainer_params["improve_start_when_dummy_ok"] and env.selected_node_list.size(-1) > (self.trainer_params["max_dummy_size"] + self.env_params["problem_size"]):
            start_sign = False
        if self.trainer_params["improve_steps"] > 0. and start_sign:
            if self.model_params["improvement_only"]: # generate random solution
                if self.trainer_params["init_sol_strategy"] != "POMO":
                    cons_reward = env.get_initial_solutions(strategy = self.trainer_params["init_sol_strategy"],
                                                            k = self.trainer_params["select_top_k"], max_dummy_size=self.trainer_params["max_dummy_size"])
                else:
                    cons_reward = self._get_pomo_initial_solution(env, data, batch_size = self.trainer_params["train_batch_size"],
                                                                  rollout_size=self.trainer_params['select_top_k'],
                                                                  eval_type="softmax" if self.trainer_params['select_top_k']>1 else "argmax",
                                                                  aug_factor=1)
                    if "TSP" not in self.args.problem: self.metric_logger.dummy_size.update(env.dummy_size, batch_size)
            else:
                cons_reward = torch.stack(reward).sum(0)
            if self.trainer_params["baseline"] == "share":
                cons_log_prob = prob_list.log().sum(dim=2) # (batch, pomo)
            elif self.trainer_params["neighborhood_search"]:
                cons_log_prob = prob_list
            else:
                cons_log_prob = None
            improve_loss, improve_reward, select_idx, best_solution, best_reward, is_improved, bl_construct_detach, bl_construct  = self._improvement(env, self.trainer_params["epsilon"], cons_reward, batch_reward, weights, cons_log_prob)
            # if "TSP" not in self.args.problem: self.metric_logger.dummy_size.update(env.dummy_size, batch_size)
        else:
            improve_loss, improve_reward, select_idx, best_solution = torch.tensor(0.0), None, None, None
        # if self.rank==0: print(f"Rank {self.rank} >> improvement time: ", time.time() - tik)

        ###########################################Step & Return########################################
        if not self.model_params["improvement_only"]:
            construct_loss = self._get_construction_output(infeasible, reward, prob_list, improve_reward, select_idx, probs_return_list, self.trainer_params["epsilon"], bl_construct_detach, bl_construct)
            # add SL loss
            if self.model_params['pip_decoder'] and self.is_train_pip_decoder:
                sl_loss_mean = sl_loss_list.mean()
                construct_loss = sl_loss_mean if construct_loss.isnan() else construct_loss + sl_loss_mean
                # Calculate the prediction accuracy
                try:
                    tn, fp, fn, tp = confusion_matrix((label_LIST > .5).astype(np.int32), (pred_LIST > .5).astype(np.int32)).ravel()
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    infsb_accuracy = tp / (fn + tp)
                    fsb_accuracy = tn / (tn + fp)
                except:
                    accuracy = 0.
                    infsb_accuracy = 0.
                    fsb_accuracy = 0.
                    tn, fp, fn, tp = 0, 0, 0, 0
        else:
            construct_loss = 0.0

        if not self.model_params["improvement_only"] and self.trainer_params["imitation_learning"] and best_solution is not None:
            env.load_problems(batch_size, rollout_size=1, problems=data, aug_factor=1)
            reset_state, _, _ = env.reset()
            state, reward, done = env.pre_step()
            imit_prob_list = torch.zeros(size=(batch_size, 1, 0)).to(self.device)
            for step in range(best_solution.size(-1)): # while not done:
                with torch.amp.autocast(device_type="cuda", enabled=amp_training):
                    _, prob, _ = self.model(state, pomo=self.env_params["pomo_start"], selected=best_solution[:,:,step],
                                                   candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None)
                if self.model_params["pip_decoder"]: prob, _ = prob
                imit_prob_list = torch.cat((imit_prob_list, prob[:, :, None]), dim=2)  # shape: (batch, pomo, solution)
                # shape: (batch, pomo)
                use_predicted_PI_mask=False
                state, reward, done, infeasible = env.step(best_solution[:,:,step].to(self.device),
                                                           out_reward=self.trainer_params["out_reward"],
                                                           soft_constrained=self.trainer_params["soft_constrained"],
                                                           backhaul_mask=self.trainer_params["backhaul_mask"],
                                                           penalty_normalize=self.trainer_params["penalty_normalize"],
                                                           generate_PI_mask=self.trainer_params["generate_PI_mask"],
                                                           use_predicted_PI_mask=use_predicted_PI_mask,
                                                           pip_step=self.trainer_params["pip_step"]
                                                           )
            imitation_loss = -(is_improved * imit_prob_list.mean(-1).mean(-1)).mean()
            self.metric_logger.construct_metrics["imitation_loss"].update(imitation_loss.item(), batch_size)
            self.metric_logger.construct_metrics["is_improved"].update(is_improved.sum()/batch_size, batch_size)

        ##########################################RE-Construction#######################################
        ##########################################RE-Construction#######################################
        ##########################################RE-Construction#######################################
        if self.args.problem == "VRPBLTW" and self.trainer_params["improve_steps"] > 0. and start_sign and self.trainer_params["reconstruct"]: # reconstruct after improvement
            # reload the environment
            env.load_problems(batch_size, rollout_size=rollout_size, problems=data, aug_factor=1)
            reset_state, _, _ = env.reset()
            state, reward, done = env.pre_step()
            # pre-forward using the best solution (batch_size, 1, solution_length)
            if "TSP" not in self.args.problem: best_solution = get_solution_with_dummy_depot(best_solution, env.problem_size)
            best_solution_rec = sol2rec(best_solution).view(batch_size, -1)
            _, context, _, _ = env.get_costs(best_solution_rec, get_context=True)
            try:
                self.model.module.pre_forward_rc(env, best_solution_rec, context)
            except:
                self.model.pre_forward_rc(env, best_solution_rec, context)

            # Initialize the prob list
            prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0)).to(self.device)
            probs_return_list = torch.zeros(size=(batch_size, env.pomo_size, env.problem_size+1, 0)).to(self.device) if self.trainer_params["probs_return"] else None
            # Start construction
            # tik = time.time()
            while not done:
                with torch.amp.autocast(device_type="cuda", enabled=amp_training):
                    selected, prob, probs_return = self.model(state, pomo=self.env_params["pomo_start"],
                                                              candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None,
                                                              return_probs=self.trainer_params["probs_return"])
                                                              # return_probs=self.trainer_params["reward_gating"])
                if probs_return is not None:
                    # constraint_mask = env.simulated_ninf_flag.to(self.device)
                    # mask_list = torch.cat((mask_list, constraint_mask[:, :, :, None]), dim=3)
                    probs_return_list = torch.cat((probs_return_list, probs_return[:, :, :, None]), dim=3)
                state, reward, done, infeasible = env.step(selected.to(self.device),
                                                           out_reward=self.trainer_params["out_reward"],
                                                           soft_constrained=self.trainer_params["soft_constrained"],
                                                           backhaul_mask=self.trainer_params["backhaul_mask"],
                                                           penalty_normalize=self.trainer_params["penalty_normalize"])
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2) # shape: (batch, pomo, solution)
            if self.trainer_params["reconstruct_improve_bonus"]:
                rc_reward = (-torch.stack(reward).sum(0)) # shape: (batch, pomo)
                improve_reward = torch.clamp_min(best_reward.view(batch_size, 1) - rc_reward, 0.0)
            reconstruct_loss = self._get_reconstruction_output(infeasible, reward, prob_list, probs_return_list, self.trainer_params["epsilon"], improve_reward)

        if self.epoch == 1: self._print_log()
        if accumulation_step == 0:
            self.model.zero_grad()
            self.optimizer.zero_grad()
        ## way 1: static weights
        coefficient = 1. if self.trainer_params["improvement_only"] else self.trainer_params["coefficient"]
        ## way 2: dynamic weights based on scale
        if self.trainer_params["dynamic_coefficient"] and start_sign:
            coefficient = find_optimal_coe(construct_loss.detach(),improve_loss.detach())
        # if self.trainer_params["dynamic_coefficient"]:
        #     coefficient = 10 ** (torch.log10(torch.abs(construct_loss.detach()/improve_loss.detach())).floor())
        #     print(">> dynamic_coefficient: {}".format(coefficient))
        ## way2.1: dynamic weights based value (Auto-Loss Balancing)
        # coefficient = torch.abs(construct_loss.detach() / (improve_loss.detach() + 1e-4))
        ## way 3: uncertainty weight
        if self.trainer_params["uncertainty_weight"]:
            loss = self.loss_fn(construct_loss, improve_loss)
            self.metric_logger.sigma1.update(self.loss_fn.sigma1)
            self.metric_logger.sigma2.update(self.loss_fn.sigma2)
        else:
            loss = construct_loss  + improve_loss * coefficient
            self.metric_logger.coefficient.update(coefficient)
            if self.trainer_params["reward_gating"] or self.trainer_params["subgradient"]:
                self.metric_logger.lambda_tw.update(self.lambda_[0].mean().item())
                self.metric_logger.lambda_demand.update(self.lambda_[1].mean().item())
                self.metric_logger.lambda_backhaul.update(self.lambda_[2].mean().item())
                self.metric_logger.lambda_dl.update(self.lambda_[3].mean().item())
                if self.trainer_params["subgradient"]:
                    _, total_timeout_reward, timeout_nodes_reward, total_out_of_dl_reward, out_of_dl_nodes_reward, total_out_of_capacity_reward, out_of_capacity_nodes_reward = reward
                    grad_timeout = total_timeout_reward + timeout_nodes_reward
                    grad_out_of_capacity = total_out_of_capacity_reward + out_of_capacity_nodes_reward
                    grad_out_of_dl = total_out_of_dl_reward + out_of_dl_nodes_reward
                    # Update Lagrange multipliers with subgradient and learning rate, ensuring non-negative values
                    self.lambda_[0] = torch.clamp(self.lambda_[0] - self.subgradient_lr * grad_timeout.mean(), min=0)
                    self.lambda_[1] = torch.clamp(self.lambda_[1] - self.subgradient_lr * grad_out_of_capacity.mean(), min=0)
                    self.lambda_[3] = torch.clamp(self.lambda_[3] - self.subgradient_lr * grad_out_of_dl.mean(), min=0)
                    # Reduce learning rate as iterations progress to ensure convergence
                    self.subgradient_lr = self.subgradient_lr / (1 + get_optimizer_step(self.optimizer))
        if not self.model_params["improvement_only"] and self.trainer_params["imitation_learning"] and best_solution is not None:
            loss += imitation_loss * self.trainer_params["imitation_loss_weight"]
        if self.args.problem == "VRPBLTW" and self.trainer_params["improve_steps"] > 0. and start_sign and self.trainer_params["reconstruct"]: # reconstruct after improvement
            loss += reconstruct_loss
        loss = loss / self.trainer_params["accumulation_steps"]
        if not amp_training:
            # if self.model_params["use_LoRA"]:
            #     # OLD IMPLEMENTATION WITH ALL THE ENCODER AS W0
            #     # only update the parameters in the large model and frozen the parameters of LoRA
            #     # with torch.autograd.set_detect_anomaly(True):
            #     loss[:self.trainer_params["LoRA_begin_step"]].mean().backward(retain_graph=True)
            #     model = self.model.module if isinstance(self.model, DDP) else self.model
            #     saved_grads = {}
            #     for name, param in model.named_parameters():
            #         if 'lora_A' in name or 'lora_B' in name:
            #             param.grad = None  # remove the gradients of LoRA
            #         else:
            #             saved_grads[name] = param.grad.clone()
            #             param.grad = None
            #             # frozen the parameters of large model and release the gradients of LoRA
            #             param.requires_grad = False # not calculate the gradients of large model later
            #     # # update the parameters until accumulating enough accumulation_steps
            #     # if accumulation_step == self.trainer_params["accumulation_steps"] - 1: self.optimizer.step()
            #     # # only update the parameters in LoRA when step > LoRA_begin_step
            #     if accumulation_step == 0: self.optimizer.zero_grad()  # clear the gradients
            #
            #     # with torch.autograd.set_detect_anomaly(True):
            #     loss[self.trainer_params["LoRA_begin_step"]:].mean().backward()
            #
            #     # use the grad recoreded before to update the large model
            #     for name, param in model.named_parameters():
            #         if 'lora_A' not in name and 'lora_B' not in name:
            #             param.grad = saved_grads[name]
            #     if accumulation_step == self.trainer_params["accumulation_steps"] - 1: self.optimizer.step()
            #
            #     # release the frozen parameters of the large model
            #     for name, param in model.named_parameters():
            #         if 'lora_A' not in name and 'lora_B' not in name:
            #             param.requires_grad = True
            # else:
            loss.backward()
            # update the parameters until accumulating enough accumulation_steps
            if accumulation_step == self.trainer_params["accumulation_steps"] - 1: self.optimizer.step()
        else:
            # with torch.autograd.set_detect_anomaly(True):
            self.scaler.scale(loss).backward(retain_graph=True)
            if accumulation_step == self.trainer_params["accumulation_steps"] - 1:
                # update the parameters until accumulating enough accumulation_steps
                self.scaler.step(self.optimizer)
                self.scaler.update()

        # update then fuse the LoRA parameters into the original parameters
        if self.model_params["use_LoRA"] and accumulation_step == self.trainer_params["accumulation_steps"] - 1:
            try:
                self.model.LoRA_fusion()
            except:
                self.model.module.LoRA_fusion()

        if self.model_params['pip_decoder'] and self.is_train_pip_decoder:
            sl_loss_out = sl_loss_list.mean().item()
            return [sl_loss_out, accuracy, infsb_accuracy, (fn + tp), fsb_accuracy, (tn + fp)]
        else:
            return None

    def _val_one_batch(self, data, env, aug_factor=1, eval_type="argmax"):
        sample_size = 1
        if self.tester_params["eval_only"]:
            sample_size = self.tester_params['sample_size'] if self.model_params['eval_type'] == "softmax" else 1

            # Sampling: augment data based on sample_size: [batch_size, ...] -> [batch_size x sample_size, ...]
            if self.model_params['eval_type'] == "softmax":
                data = list(data)
                for i, d in enumerate(data):
                    if d.dim() == 1:
                        data[i] = d.repeat(sample_size)
                    elif d.dim() == 2:
                        data[i] = d.repeat(sample_size, 1)
                    elif d.dim() == 3:
                        data[i] = d.repeat(sample_size, 1, 1)
        if self.model_params["pip_decoder"]:
            pred_LIST, label_LIST = np.array([]), np.array([])

        self.model.eval()
        try:
            self.model.module.set_eval_type(eval_type)
        except:
            self.model.set_eval_type(eval_type)

        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        if self.model_params["polynet"]:
            z_sample_size = self.trainer_params['val_z_sample_size']
            z_dim = self.model_params['z_dim']
            if self.env_params['pomo_start']:
                starting_points = self.env_params['problem_size']
                rollout_size = starting_points * z_sample_size
            else:
                starting_points = 1
                rollout_size = z_sample_size
        else:
            rollout_size = self.env_params["pomo_size"]

        if self.tester_params["is_lib"]:
            rollout_size=env.problem_size
            # self.trainer_params["select_top_k"] = env.problem_size
        print("rollout_size", rollout_size)

        with torch.no_grad():
            if not (self.trainer_params["improvement_only"] and self.trainer_params["val_init_sol_strategy"] == "POMO"):
                env.load_problems(batch_size, rollout_size=rollout_size, problems=data, aug_factor=aug_factor)
                reset_state, _, _ = env.reset()

                state, reward, done = env.pre_step()
            ###########################################Construction########################################
            if not self.model_params["improvement_only"]:
                if self.rank == 0: tik = time.time()
                z = self.sample_z_vectors(batch_size * aug_factor, starting_points, z_dim, z_sample_size, rollout_size)  if self.model_params["polynet"] else None
                try:
                    self.model.module.pre_forward(reset_state, z)
                    if self.model_params["pip_decoder"] and (self.lazy_model is not None) and (not self.is_train_pip_decoder):
                        # ATTENTION: only use the predicted mask for validation when not training?
                        self.lazy_model.module.eval()
                        self.lazy_model.pre_forward(reset_state)
                except:
                    self.model.pre_forward(reset_state, z)
                    if self.model_params["pip_decoder"] and (self.lazy_model is not None) and (not self.is_train_pip_decoder):
                        # ATTENTION: only use the predicted mask for validation when not training?
                        self.lazy_model.eval()
                        self.lazy_model.pre_forward(reset_state)
                while not done:
                    use_predicted_PI_mask = True if (self.model_params['pip_decoder'] and not self.is_train_pip_decoder) else False
                    # print(use_predicted_PI_mask)
                    if self.model_params["pip_decoder"] and self.lazy_model is not None and not (self.is_train_pip_decoder) and env.selected_count >= 1:
                        use_predicted_PI_mask = self.lazy_model(state, candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None, use_predicted_PI_mask=False, no_select_prob=True)
                    selected, prob, _ = self.model(state, pomo=self.env_params["pomo_start"],
                                                candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None,
                                                use_predicted_PI_mask=use_predicted_PI_mask)
                    # shape: (batch, pomo)
                    if self.model_params['pip_decoder']:
                        _, probs_sl = prob
                        if self.model_params['pip_decoder'] and (env.selected_count >= 1) and (env.selected_count < env.problem_size - 1):
                            label = torch.where(env.simulated_ninf_flag == float('-inf'), 1., env.simulated_ninf_flag)
                            visited_mask = (env.visited_ninf_flag == float('-inf'))
                            label = label[~visited_mask]
                            probs_sl = probs_sl[~visited_mask]
                            pred_LIST = np.append(pred_LIST, probs_sl.detach().cpu().numpy())
                            label_LIST = np.append(label_LIST, label.detach().cpu().numpy())
                    # ATTENTION: PIP-D always generate PI mask during validation
                    generate_PI_mask = True if self.model_params['pip_decoder'] else self.trainer_params["generate_PI_mask"]
                    # print(generate_PI_mask)
                    use_predicted_PI_mask = ((not isinstance(use_predicted_PI_mask, bool) or use_predicted_PI_mask == True) or not self.trainer_params["use_real_PI_mask"])
                    state, reward, done, infeasible = env.step(selected,
                                                               out_reward = (self.tester_params["best_solution_path"] is not None),
                                                               soft_constrained = self.trainer_params["soft_constrained"],
                                                               backhaul_mask = self.trainer_params["backhaul_mask"],
                                                               generate_PI_mask=generate_PI_mask,
                                                               use_predicted_PI_mask=use_predicted_PI_mask,
                                                               pip_step=self.trainer_params["pip_step"]
                                                               )
                # Return
                aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible, best_solution = self._get_construction_output_val(aug_factor * sample_size, infeasible, reward, env.selected_node_list)
                if self.rank==0: print(f"Rank {self.rank} >> val construction time: ", time.time() - tik)
            ###########################################Improvement########################################
            if self.trainer_params["validation_improve_steps"] > 0.:
                if self.model_params["clean_cache"]: torch.cuda.empty_cache()
                # self.model.decoder.k = None
                # self.model.decoder.v = None
                if self.rank==0: tik = time.time()
                if self.model_params["improvement_only"]:  # generate random solution
                    if self.trainer_params["val_init_sol_strategy"] != "POMO":
                        env.get_initial_solutions(strategy=self.trainer_params["val_init_sol_strategy"], k=self.env_params["pomo_size"], max_dummy_size=self.trainer_params["max_dummy_size"])
                        self._get_construction_output_val(aug_factor, env.infeasible, env._get_travel_distance(), env.selected_node_list)
                    else:
                        self._get_pomo_initial_solution(env, data, batch_size, rollout_size, eval_type="argmax", aug_factor=aug_factor, val=True)
                # FIXME: should we use cost+penalty?
                aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible, best_solution, is_improved = self._val_improvement(env, aug_factor * sample_size, -env._get_travel_distance(), best_solution=best_solution if self.tester_params["best_solution_path"] is not None else None) # after construction and improvement
                if self.rank==0: print(f"Rank {self.rank} >> val improvement time: ", time.time() - tik)
            ##########################################RE-Construction#######################################
            if self.args.problem == "VRPBLTW" and self.trainer_params["validation_improve_steps"] > 0. and self.trainer_params["reconstruct"]:  # reconstruct after improvement
                if self.rank == 0: tik = time.time()
                # reconstruct the solution (env already load problem)
                reset_state, _, _ = env.reset()
                state, reward, done = env.pre_step()
                # pre-forward using the best solution (batch_size, 1, solution_length)
                if "TSP" not in self.args.problem: best_solution = get_solution_with_dummy_depot(best_solution, env.problem_size)
                best_solution_rec = sol2rec(best_solution).view(batch_size, -1).repeat(aug_factor,1)
                _, context, _, _ = env.get_costs(best_solution_rec, get_context=True)
                try:
                    self.model.module.pre_forward_rc(env, best_solution_rec, context)
                except:
                    self.model.pre_forward_rc(env, best_solution_rec, context)
                while not done:
                    selected, _, _ = self.model(state, pomo=self.env_params["pomo_start"], candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None)
                    # shape: (batch, pomo)
                    state, reward, done, infeasible = env.step(selected, soft_constrained = self.trainer_params["soft_constrained"], backhaul_mask = self.trainer_params["backhaul_mask"])
                # Retur
                aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible = self._get_reconstruction_output_val(aug_factor, infeasible, reward, aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible)
                if self.rank == 0: print(f"Rank {self.rank} >> val reconstruction time: ", time.time() - tik)
            ###########################################RE-Improvement########################################
            if self.trainer_params["val_reconstruct_times"] > 1.:
                if self.model_params["clean_cache"]: torch.cuda.empty_cache()
                # self.model.decoder.k = None
                # self.model.decoder.v = None
                if self.rank==0: tik = time.time()
                if self.model_params["improvement_only"]:  # generate random solution
                    if self.trainer_params["val_init_sol_strategy"] != "POMO":
                        env.get_initial_solutions(strategy=self.trainer_params["val_init_sol_strategy"], k=self.env_params["pomo_size"], max_dummy_size=self.trainer_params["max_dummy_size"])
                        self._get_construction_output_val(aug_factor, env.infeasible, env._get_travel_distance())
                    else:
                        self._get_pomo_initial_solution(env, data, batch_size, rollout_size, eval_type="argmax", aug_factor=aug_factor, val=True)
                env.dummy_xy = None
                aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible, best_solution, is_improved = self._val_reimprovement(env, aug_factor, aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible) # after construction and improvement
                if self.rank==0: print(f"Rank {self.rank} >> val reimprovement time: ", time.time() - tik)
            #############################Use mask to construct the solutions again###################################
            if self.model_params["problem"] == "VRPBLTW" and self.tester_params["aux_mask"]:
                if self.rank==0: tik = time.time()
                # reconstruct the solution (env already load problem)
                reset_state, _, _ = env.reset()
                state, reward, done = env.pre_step()
                z = self.sample_z_vectors(batch_size * aug_factor, starting_points, z_dim, z_sample_size, rollout_size) if self.model_params["polynet"] else None
                if not self.model_params["improvement_only"]:
                    try:
                        self.model.module.pre_forward(reset_state, z)
                    except:
                        self.model.pre_forward(reset_state, z)
                    while not done:
                        selected, _, _ = self.model(state, pomo=self.env_params["pomo_start"], candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None)
                        # shape: (batch, pomo)
                        state, reward, done, infeasible = env.step(selected, soft_constrained = False, backhaul_mask = "hard")
                else:
                    try:
                        self.pomo_model.module.pre_forward(reset_state, z)
                    except:
                        self.pomo_model.pre_forward(reset_state, z)
                    while not done:
                        selected, _, _ = self.pomo_model(state, pomo=self.env_params["pomo_start"], candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None)
                        # shape: (batch, pomo)
                        state, reward, done, infeasible = env.step(selected, soft_constrained = False, backhaul_mask = "hard")
                # Obtain the minimal feasible reward
                self._supplement_construction(aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible, aug_factor, infeasible, reward)
                if self.rank==0: print(f"Rank {self.rank} >> val reconstruction time [w. mask]: ", time.time() - tik)

        if self.model_params["pip_decoder"]:
            return [pred_LIST, label_LIST]
        else:
            return None

    def _val_one_batch_with_eas(self, data, env, aug_factor=1, eval_type="argmax"):
        sample_size = 1
        if self.tester_params["eval_only"]:
            sample_size = self.tester_params['sample_size'] if self.model_params['eval_type'] == "softmax" else 1

            # Sampling: augment data based on sample_size: [batch_size, ...] -> [batch_size x sample_size, ...]
            if self.model_params['eval_type'] == "softmax":
                data = list(data)
                for i, d in enumerate(data):
                    if d.dim() == 1:
                        data[i] = d.repeat(sample_size)
                    elif d.dim() == 2:
                        data[i] = d.repeat(sample_size, 1)
                    elif d.dim() == 3:
                        data[i] = d.repeat(sample_size, 1, 1)
        if self.model_params["pip_decoder"]:
            pred_LIST, label_LIST = np.array([]), np.array([])
        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        rollout_size = self.env_params["pomo_size"]
        if self.tester_params["is_lib"]:
            rollout_size = env.problem_size
            # self.trainer_params["select_top_k"] = env.problem_size
        # rollout_size += 1
        print("rollout_size", rollout_size)
        iterations = self.tester_params['EAS_params']['iterations']
        enable_EAS = self.tester_params['EAS_params']['enable']

        self.model.decoder.reset_EAS_layers(batch_size * aug_factor)  # initialize/reset EAS layers
        EAS_layer_parameters = self.model.decoder.get_EAS_parameters()
        # Only store gradients for new EAS layer weights
        self.model.requires_grad_(False)
        for t in EAS_layer_parameters:
            t.requires_grad_(True)
        optimizer = Optimizer(EAS_layer_parameters, lr=self.tester_params['EAS_params']['lr'])
        self.model.train()

        if not (self.trainer_params["improvement_only"] and self.trainer_params["val_init_sol_strategy"] == "POMO"):
            env.load_problems(batch_size, rollout_size=rollout_size, problems=data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()

            state, reward, done = env.pre_step()
        ###########################################Construction########################################
        if not self.model_params["improvement_only"]:
            if self.rank == 0: tik = time.time()
            z = self.sample_z_vectors(batch_size * aug_factor, starting_points, z_dim, z_sample_size, rollout_size) if self.model_params["polynet"] else None
            try:
                self.model.module.pre_forward(reset_state, z)
                if self.model_params["pip_decoder"] and (self.lazy_model is not None) and (
                not self.is_train_pip_decoder):
                    # ATTENTION: only use the predicted mask for validation when not training?
                    self.lazy_model.module.eval()
                    self.lazy_model.pre_forward(reset_state)
            except:
                self.model.pre_forward(reset_state, z)
                if self.model_params["pip_decoder"] and (self.lazy_model is not None) and (
                not self.is_train_pip_decoder):
                    # ATTENTION: only use the predicted mask for validation when not training?
                    self.lazy_model.eval()
                    self.lazy_model.pre_forward(reset_state)
            incumbent_reward = torch.ones(batch_size).float() * float('-inf')
            incumbent_solution = None
            for iter in tqdm(range(iterations)):
                if incumbent_solution is not None:
                    env.reset_pomo_size_for_eas(rollout_size+1)
                env.reset()
                prob_list = torch.zeros(size=(batch_size * aug_factor, env.pomo_size, 0))
                # POMO Rollout
                ###############################################
                state, reward, done = env.pre_step()
                while not done:

                    if incumbent_solution is not None:
                        incumbent_action = incumbent_solution[:, env.selected_count]
                    else:
                        incumbent_action = None

                    use_predicted_PI_mask = True if (self.model_params['pip_decoder'] and not self.is_train_pip_decoder) else False
                    # print(use_predicted_PI_mask)
                    if self.model_params["pip_decoder"] and self.lazy_model is not None and not (self.is_train_pip_decoder) and env.selected_count >= 1:
                        use_predicted_PI_mask = self.lazy_model(state,
                                                                candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None,
                                                                use_predicted_PI_mask=False, no_select_prob=True)
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        selected, prob, _ = self.model(state, pomo=self.env_params["pomo_start"],
                                                       candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None,
                                                       use_predicted_PI_mask=use_predicted_PI_mask, EAS_incumbent_action=incumbent_action)
                    # shape: (batch, pomo)

                    if self.model_params['pip_decoder']:
                        _, probs_sl = prob
                        if self.model_params['pip_decoder'] and (env.selected_count >= 1) and (
                                env.selected_count < env.problem_size - 1):
                            label = torch.where(env.simulated_ninf_flag == float('-inf'), 1., env.simulated_ninf_flag)
                            visited_mask = (env.visited_ninf_flag == float('-inf'))
                            label = label[~visited_mask]
                            probs_sl = probs_sl[~visited_mask]
                            pred_LIST = np.append(pred_LIST, probs_sl.detach().cpu().numpy())
                            label_LIST = np.append(label_LIST, label.detach().cpu().numpy())
                    # ATTENTION: PIP-D always generate PI mask during validation
                    generate_PI_mask = True if self.model_params['pip_decoder'] else self.trainer_params["generate_PI_mask"]
                    # print(generate_PI_mask)
                    use_predicted_PI_mask = ((not isinstance(use_predicted_PI_mask, bool) or use_predicted_PI_mask == True) or not self.trainer_params["use_real_PI_mask"])

                    state, reward, done, infeasible = env.step(selected,
                                                               # out_reward=self.trainer_params["out_reward"],
                                                               soft_constrained=self.trainer_params["soft_constrained"],
                                                               backhaul_mask=self.trainer_params["backhaul_mask"],
                                                               generate_PI_mask=generate_PI_mask,
                                                               use_predicted_PI_mask=use_predicted_PI_mask,
                                                               pip_step=self.trainer_params["pip_step"]
                                                               )
                    prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

                # Incumbent solution
                ###############################################
                # CVRP have all feasible solutions
                # if isinstance(reward, list):
                #     inf = torch.tensor(float('inf'))
                #     feasible_lengths = torch.where(reward[2] == 0, -reward[0], inf)  # shape: (B, P)
                #     indices_feasible = torch.argmin(feasible_lengths, dim=1)  # shape: (B,)
                #     indices_violation = torch.argmin(reward[2], dim=1)  # shape: (B,)
                #     has_feasible = (reward[2] == 0).any(dim=1)  # shape: (B,)
                #     selected_indices = torch.where(has_feasible, indices_feasible, indices_violation)
                #     selected_lengths = (reward[0]+reward[1]+reward[2]).gather(dim=1, index=selected_indices.unsqueeze(1)).squeeze(1)
                max_reward, max_idx = reward.max(dim=1)  # get best results from rollouts + Incumbent
                # shape: (aug_batch,)
                incumbent_reward = max_reward

                gathering_index = max_idx[:, None, None].expand(-1, 1, env.selected_count)
                new_incumbent_solution = env.selected_node_list.gather(dim=1, index=gathering_index)
                new_incumbent_solution = new_incumbent_solution.squeeze(dim=1)
                # shape: (aug_batch, tour_len)

                solution_max_length = 1000
                incumbent_solution = torch.zeros(size=(batch_size * aug_factor, solution_max_length), dtype=torch.long)
                incumbent_solution[:, :env.selected_count] = new_incumbent_solution

                # Loss: POMO RL
                ###############################################
                pomo_prob_list = prob_list[:, :-1, :]
                # shape: (aug_batch, pomo, tour_len)
                pomo_reward = reward[:, :-1]
                # shape: (aug_batch, pomo)

                advantage = pomo_reward - pomo_reward.mean(dim=1, keepdim=True)
                # shape: (aug_batch, pomo)
                log_prob = pomo_prob_list.log().sum(dim=2)
                # size = (aug_batch, pomo)
                loss_RL = -advantage * log_prob  # Minus Sign: To increase REWARD

                # shape: (aug_batch, pomo)
                loss_RL = loss_RL.mean(dim=1)
                # shape: (aug_batch,)

                # Loss: IL
                ###############################################
                imitation_prob_list = prob_list[:, -1, :]
                # shape: (aug_batch, tour_len)
                log_prob = imitation_prob_list.log().sum(dim=1)
                # shape: (aug_batch,)
                loss_IL = -log_prob  # Minus Sign: to increase probability
                # shape: (aug_batch,)

                # Back Propagation
                ###############################################
                optimizer.zero_grad()

                loss = loss_RL + self.tester_params['EAS_params']['lambda'] * loss_IL
                # shape: (aug_batch,)
                loss.sum().backward()

                optimizer.step()

            # Return
            aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible, best_solution = self._get_construction_output_val(
                aug_factor * sample_size, infeasible, pomo_reward, env.selected_node_list, incumbent_reward=incumbent_reward)
            if self.rank == 0: print(f"Rank {self.rank} >> val construction time: ", time.time() - tik)
        ###########################################Improvement########################################
        if self.trainer_params["validation_improve_steps"] > 0.:
            if self.model_params["clean_cache"]: torch.cuda.empty_cache()
            # self.model.decoder.k = None
            # self.model.decoder.v = None
            if self.rank == 0: tik = time.time()
            if self.model_params["improvement_only"]:  # generate random solution
                if self.trainer_params["val_init_sol_strategy"] != "POMO":
                    env.get_initial_solutions(strategy=self.trainer_params["val_init_sol_strategy"],
                                              k=self.env_params["pomo_size"],
                                              max_dummy_size=self.trainer_params["max_dummy_size"])
                    self._get_construction_output_val(aug_factor, env.infeasible, env._get_travel_distance(),
                                                      env.selected_node_list)
                else:
                    self._get_pomo_initial_solution(env, data, batch_size, rollout_size, eval_type="argmax",
                                                    aug_factor=aug_factor, val=True)
            aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible, best_solution, is_improved = self._val_improvement_with_eas(
                env, aug_factor * sample_size, -env._get_travel_distance(), epsilon=self.trainer_params["epsilon"])  # after construction and improvement
            if self.rank == 0: print(f"Rank {self.rank} >> val improvement time: ", time.time() - tik)
        ##########################################RE-Construction#######################################
        if self.args.problem == "VRPBLTW" and self.trainer_params["validation_improve_steps"] > 0. and \
                self.trainer_params["reconstruct"]:  # reconstruct after improvement
            if self.rank == 0: tik = time.time()
            # reconstruct the solution (env already load problem)
            reset_state, _, _ = env.reset()
            state, reward, done = env.pre_step()
            # pre-forward using the best solution (batch_size, 1, solution_length)
            if "TSP" not in self.args.problem: best_solution = get_solution_with_dummy_depot(best_solution, env.problem_size)
            best_solution_rec = sol2rec(best_solution).view(batch_size, -1).repeat(aug_factor, 1)
            _, context, _, _ = env.get_costs(best_solution_rec, get_context=True)
            try:
                self.model.module.pre_forward_rc(env, best_solution_rec, context)
            except:
                self.model.pre_forward_rc(env, best_solution_rec, context)
            while not done:
                selected, _, _ = self.model(state, pomo=self.env_params["pomo_start"],
                                            candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None)
                # shape: (batch, pomo)
                state, reward, done, infeasible = env.step(selected,
                                                           soft_constrained=self.trainer_params["soft_constrained"],
                                                           backhaul_mask=self.trainer_params["backhaul_mask"])
            # Retur
            aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible = self._get_reconstruction_output_val(
                aug_factor, infeasible, reward, aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible)
            if self.rank == 0: print(f"Rank {self.rank} >> val reconstruction time: ", time.time() - tik)
        ###########################################RE-Improvement########################################
        if self.trainer_params["val_reconstruct_times"] > 1.:
            if self.model_params["clean_cache"]: torch.cuda.empty_cache()
            # self.model.decoder.k = None
            # self.model.decoder.v = None
            if self.rank == 0: tik = time.time()
            if self.model_params["improvement_only"]:  # generate random solution
                if self.trainer_params["val_init_sol_strategy"] != "POMO":
                    env.get_initial_solutions(strategy=self.trainer_params["val_init_sol_strategy"],
                                              k=self.env_params["pomo_size"],
                                              max_dummy_size=self.trainer_params["max_dummy_size"])
                    self._get_construction_output_val(aug_factor, env.infeasible, env._get_travel_distance())
                else:
                    self._get_pomo_initial_solution(env, data, batch_size, rollout_size, eval_type="argmax",
                                                    aug_factor=aug_factor, val=True)
            env.dummy_xy = None
            aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible, best_solution, is_improved = self._val_reimprovement(
                env, aug_factor, aug_score_fsb, no_aug_score_fsb, aug_feasible,
                no_aug_feasible)  # after construction and improvement
            if self.rank == 0: print(f"Rank {self.rank} >> val reimprovement time: ", time.time() - tik)
        #############################Use mask to construct the solutions again###################################
        if self.model_params["problem"] == "VRPBLTW" and self.tester_params["aux_mask"]:
            if self.rank == 0: tik = time.time()
            # reconstruct the solution (env already load problem)
            reset_state, _, _ = env.reset()
            state, reward, done = env.pre_step()
            if not self.model_params["improvement_only"]:
                z = self.sample_z_vectors(batch_size * aug_factor, starting_points, z_dim, z_sample_size,
                                          rollout_size) if self.model_params["polynet"] else None
                try:
                    self.model.module.pre_forward(reset_state, z)
                except:
                    self.model.pre_forward(reset_state, z)
                while not done:
                    selected, _, _ = self.model(state, pomo=self.env_params["pomo_start"],
                                                candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None)
                    # shape: (batch, pomo)
                    state, reward, done, infeasible = env.step(selected, soft_constrained=False,
                                                               backhaul_mask="hard")
                # Obtain the minimal feasible reward
                self._supplement_construction(aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible,
                                              aug_factor, infeasible, reward)
            if self.rank == 0: print(f"Rank {self.rank} >> val reconstruction time [w. mask]: ", time.time() - tik)

        if self.model_params["pip_decoder"]:
            return [pred_LIST, label_LIST]
        else:
            return None

    def _val_and_stat(self, dir, val_path, env, batch_size=500, val_episodes=1000, compute_gap=False, epoch=1):
        if self.model_params["pip_decoder"]:
            pred_LIST, label_LIST = np.array([]), np.array([])
        if self.model_params["pip_decoder"] and (self.lazy_model is not None) and (not self.is_train_pip_decoder):
            if self.rank == 0: print(">> Use PIP-D predicted mask for validation...")
        elif self.trainer_params["use_real_PI_mask"] and self.model_params["generate_PI_mask"]:
            if self.rank == 0: print(">> Use PI masking for validation...")
        self.val_metric_logger = val_metric_logger(self)

        # gap may get wrong, not synchronized!!
        # assert batch_size % self.world_size == 0
        # data = env.load_dataset(os.path.join(dir, val_path), offset=0, num_samples=bs)
        # val_sampler = DistributedSampler(data, num_replicas=self.world_size, rank=self.rank)
        # val_loader = DataLoader(dataset=data, batch_size=batch_size // self.world_size, shuffle=False, sampler=val_sampler)
        #
        # with torch.no_grad():
        #     for batch in val_loader:
        #         batch = batch.to(self.rank)
        #         self._val_one_batch(batch, env, aug_factor=8, eval_type="argmax")
        if self.tester_params["eval_only"]: self.time_estimator.reset()
        episode = 0

        while episode < val_episodes:
            remaining = val_episodes - episode
            bs = min(batch_size, remaining)
            path_ = os.path.join(dir, val_path) if not self.tester_params["is_lib"] else val_path
            data = env.load_dataset(path_, offset=episode, num_samples=bs)
            print(self.model_params["eval_type"])
            # env_params = {'problem_size': node_xy.size(1), 'pomo_size': node_xy.size(1), 'loc_scaler': loc_scaler
            env.pomo_size = data[2].size(1)
            env.problem_size = data[2].size(1)
            env.loc_scaler = 1000

            if self.tester_params['EAS_params']['enable']:
                output = self._val_one_batch_with_eas(data, env, aug_factor=8, eval_type=self.model_params["eval_type"])
            else:
                output = self._val_one_batch(data, env, aug_factor=8, eval_type=self.model_params["eval_type"])
            if output is not None:
                pred_LIST = np.append(pred_LIST, output[0])
                label_LIST = np.append(label_LIST, output[1])

            if self.tester_params['best_solution_path'] is not None:
                if episode == 0:
                    self.val_metric_logger.best_solution_all = self.val_metric_logger.best_solution
                    self.val_metric_logger.best_reward_all = self.val_metric_logger.best_reward
                    self.val_metric_logger.best_feasible_all = self.val_metric_logger.best_feasible
                    self.val_metric_logger.feasible_bsf_history = self.val_metric_logger.refinement_feasible_bsf_history # 0~batch, 2+T
                    self.val_metric_logger.reward_history = self.val_metric_logger.refinement_reward_bsf_history # 0~batch, 2+T
                    self.val_metric_logger.solution_history = self.val_metric_logger.refinement_solution_history # 0~batch, 2+T, solution_size
                else:
                    self.val_metric_logger.best_solution_all = pad_and_cat_dim([self.val_metric_logger.best_solution_all, self.val_metric_logger.best_solution])
                    self.val_metric_logger.best_reward_all = torch.cat([self.val_metric_logger.best_reward_all, self.val_metric_logger.best_reward], dim=0)
                    self.val_metric_logger.best_feasible_all = torch.cat([self.val_metric_logger.best_feasible_all, self.val_metric_logger.best_feasible], dim=0)
                    self.val_metric_logger.feasible_bsf_history = torch.cat([self.val_metric_logger.feasible_bsf_history, self.val_metric_logger.refinement_feasible_bsf_history], dim=0)
                    self.val_metric_logger.reward_history = torch.cat([self.val_metric_logger.reward_history, self.val_metric_logger.refinement_reward_bsf_history], dim=0)
                    self.val_metric_logger.solution_history = pad_and_cat_dim([self.val_metric_logger.solution_history, self.val_metric_logger.refinement_solution_history])
            episode += bs


            if self.tester_params["eval_only"]:
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, val_episodes)
                print("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}]".format(episode, val_episodes, elapsed_time_str, remain_time_str))

        if self.model_params["pip_decoder"]:
            tn, fp, fn, tp = confusion_matrix((label_LIST > 0.5).astype(np.int32),(pred_LIST > 0.5).astype(np.int32)).ravel()
            # tn, fp, fn, tp = confusion_matrix((labels > 0.5).int().cpu(), (F.sigmoid(predict_out) > 0.5).int().cpu()).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            infsb_accuracy = tp / (fn + tp)
            fsb_accuracy = tn / (tn + fp)
            self.val_metric_logger.construct_metrics["accuracy"] = accuracy
            self.val_metric_logger.construct_metrics["infsb_accuracy"] = infsb_accuracy
            self.val_metric_logger.construct_metrics["fsb_accuracy"] = fsb_accuracy
            self.val_metric_logger.construct_metrics["infsb_sample_nums"] = (label_LIST > 0.5).astype(np.int32).sum()
            self.val_metric_logger.construct_metrics["fsb_sample_nums"] = (label_LIST < 0.5).astype(np.int32).sum()
            if self.rank == 0: print("PIP-D Validation, Auc: {:.4f}, Infeasible Auc: {:.4f} ({}), Feasible Auc: {:.4f} ({})".format(accuracy, infsb_accuracy,(fn + tp), fsb_accuracy,(tn + fp)))
        self.val_metric_logger._log_output(self)

        try:
            sol_path = get_opt_sol_path(dir, env.problem, data[1].size(1))
        except:
            sol_path = os.path.join(dir, "lkh_" + val_path)

        compute_gap = os.path.exists(sol_path) if not self.tester_params["is_lib"] else False

        if compute_gap:
            opt_sol = load_dataset(sol_path, disable_print=True)[: val_episodes]
            grid_factor = 100. if self.args.problem == "TSPTW" else 1.
            opt_sol = torch.tensor([i[0]/grid_factor for i in opt_sol])
            self.val_metric_logger._calculate_gap(self, opt_sol)
            try:
                if self.rank==0: print(f'Rank {self.rank} >> Val Score on {val_path}: [Construction] NO_AUG_Score: {self.val_metric_logger.construct_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.construct_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.construct_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.construct_metrics["aug_gap_list"]}%; Infeasible rate: {self.val_metric_logger.construct_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.construct_metrics["ins_infeasible_rate_list"]}% (instance-level)')
                if self.rank==0 and self.trainer_params["validation_improve_steps"] > 0.: print(f'Rank {self.rank} >> Val Score on {val_path}: [Improvement] NO_AUG_Score: {self.val_metric_logger.improve_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.improve_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.improve_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.improve_metrics["aug_gap_list"]}%; Infeasible rate: {self.val_metric_logger.improve_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.improve_metrics["ins_infeasible_rate_list"]}% (instance-level)')
                if self.rank==0 and self.args.problem == "VRPBLTW" and self.trainer_params["validation_improve_steps"] > 0. and self.trainer_params["reconstruct"]: print(f'Rank {self.rank} >> Val Score on {val_path}: [w/o mask] NO_AUG_Score: {self.val_metric_logger.reconstruct_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.reconstruct_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.reconstruct_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.reconstruct_metrics["aug_gap_list"]}%; Infeasible rate: {self.val_metric_logger.reconstruct_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.reconstruct_metrics["ins_infeasible_rate_list"]}% (instance-level)')
                if self.rank == 0 and self.trainer_params["val_reconstruct_times"] > 1.: print(f'Rank {self.rank} >> Val Score on {val_path}: [RE-Improvement] NO_AUG_Score: {self.val_metric_logger.reimprove_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.reimprove_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.reimprove_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.reimprove_metrics["aug_gap_list"]}%; Infeasible rate: {self.val_metric_logger.reimprove_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.reimprove_metrics["ins_infeasible_rate_list"]}% (instance-level)')
                if self.rank==0 and self.tester_params["aux_mask"]: print(f'Rank {self.rank} >> Val Score on {val_path}: [w. mask] NO_AUG_Score: {self.val_metric_logger.reconstruct_masked_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.reconstruct_masked_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.reconstruct_masked_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.reconstruct_masked_metrics["aug_gap_list"]}%; Infeasible rate: {self.val_metric_logger.reconstruct_masked_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.reconstruct_masked_metrics["ins_infeasible_rate_list"]}% (instance-level)')
            except:
                if self.rank==0: print(f'Rank {self.rank} >> Val Score on {val_path}: [Construction] NO_AUG_Score: {self.val_metric_logger.construct_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.construct_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.construct_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.construct_metrics["aug_gap_list"]}%')
                if self.rank == 0 and self.trainer_params["validation_improve_steps"] > 0.: print(f'Rank {self.rank} >> Val Score on {val_path}: [Improvement] NO_AUG_Score: {self.val_metric_logger.improve_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.improve_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.improve_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.improve_metrics["aug_gap_list"]}%')
                if self.rank==0 and self.args.problem == "VRPBLTW" and self.trainer_params["validation_improve_steps"] > 0. and self.trainer_params["reconstruct"]: print(f'Rank {self.rank} >> Val Score on {val_path}: [w/o mask] NO_AUG_Score: {self.val_metric_logger.reconstruct_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.reconstruct_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.reconstruct_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.reconstruct_metrics["aug_gap_list"]}%')
                if self.rank == 0 and self.tester_params["aux_mask"]: print(f'Rank {self.rank} >> Val Score on {val_path}: [w. mask] NO_AUG_Score: {self.val_metric_logger.reconstruct_masked_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.reconstruct_masked_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.reconstruct_masked_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.reconstruct_masked_metrics["aug_gap_list"]}%')
        else:
            if self.rank==0: print(f'Rank {self.rank} >> Val Score on {val_path}: [Construction] NO_AUG_Score: {self.val_metric_logger.construct_metrics["no_aug_score_list"]}, --> AUG_Score: {self.val_metric_logger.construct_metrics["aug_score_list"]}; Infeasible rate: {self.val_metric_logger.construct_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.construct_metrics["ins_infeasible_rate_list"]}% (instance-level)')
            if self.rank==0 and self.trainer_params["validation_improve_steps"] > 0.: print(f'Rank {self.rank} >> Val Score on {val_path}: [Improvement] NO_AUG_Score: {self.val_metric_logger.improve_metrics["no_aug_score_list"]}, --> AUG_Score: {self.val_metric_logger.improve_metrics["aug_score_list"]}; Infeasible rate: {self.val_metric_logger.improve_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.improve_metrics["ins_infeasible_rate_list"]}% (instance-level)')
            if self.rank == 0 and self.args.problem == "VRPBLTW" and self.trainer_params["validation_improve_steps"] > 0. and self.trainer_params["reconstruct"]: print(f'Rank {self.rank} >> Val Score on {val_path}: [w/o mask] NO_AUG_Score: {self.val_metric_logger.reconstruct_metrics["no_aug_score_list"]} --> AUG_Score: {self.val_metric_logger.reconstruct_metrics["aug_score_list"]}; Infeasible rate: {self.val_metric_logger.reconstruct_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.reconstruct_metrics["ins_infeasible_rate_list"]}% (instance-level)')
            if self.rank==0 and self.tester_params["aux_mask"]: print(f'Rank {self.rank} >> Val Score on {val_path}: [w. mask] NO_AUG_Score: {self.val_metric_logger.reconstruct_masked_metrics["no_aug_score_list"]}, --> AUG_Score: {self.val_metric_logger.reconstruct_masked_metrics["aug_score_list"]}; Infeasible rate: {self.val_metric_logger.reconstruct_masked_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.reconstruct_masked_metrics["ins_infeasible_rate_list"]}% (instance-level)')

    def _improvement(self, env, epsilon=None, cons_reward=None, batch_reward=None, weights=None, cons_log_prob=None):
        amp_training = self.trainer_params['amp_training']
        # solution/rec shape: (batch, K, solution)
        solution = env.selected_node_list.clone() # shape: (batch, pomo, solution)
        if not self.trainer_params["improvement_only"]: # if yes, already generate k solutions for each instance
            # select top k
            if self.trainer_params["select_top_k"] < env.pomo_size:
                solution, select_idx = select4improve(solution, cons_reward, strategy=self.trainer_params["select_strategy"],
                                                      K=self.trainer_params["select_top_k"], rnd_prob=self.trainer_params["stochastic_probability"],
                                                      diversity=self.trainer_params["diversity"])
                # solution shape: (batch, k, solution); solution_idx shape: (batch, k)
                feasibility_history = torch.gather(~env.infeasible, 1, select_idx)
                if self.trainer_params["neighborhood_search"]:
                    cons_log_prob = cons_log_prob[torch.arange(env.batch_size)[:, None], select_idx]
                    _, unconfident_indices = torch.topk(cons_log_prob, k=self.trainer_params["k_unconfident"], dim=-1, largest=False)
            else:
                # just for testing
                select_idx = torch.arange(env.pomo_size)[None, :].repeat(env.batch_size, 1)
                feasibility_history = ~env.infeasible
                _, topk2 = select4improve(solution, cons_reward, strategy=self.trainer_params["select_strategy"],
                                          K=5, rnd_prob=self.trainer_params["stochastic_probability"],
                                          diversity=self.trainer_params["diversity"])
        else:
            select_idx = torch.arange(env.pomo_size)[None, :].repeat(env.batch_size, 1)
            feasibility_history = ~env.infeasible

        if "TSP" not in self.args.problem: solution = get_solution_with_dummy_depot(solution, env.problem_size)
        batch_size, k, solution_size = solution.size()
        rec = sol2rec(solution).view(batch_size * k, -1)

        # preapare input
        obj, context, out_penalty, out_node_penalty = env.get_costs(rec, get_context=True, out_reward=self.trainer_params["out_reward"], penalty_factor = self.lambda_, seperate_obj_penalty=self.trainer_params["seperate_obj_penalty"], wo_node_penalty=self.trainer_params["wo_node_penalty"], wo_tour_penalty=self.trainer_params["wo_tour_penalty"])
        if self.trainer_params["seperate_obj_penalty"]:
            obj, penalty = obj
            obj = torch.cat((obj[:, None], obj[:, None], obj[:, None]), -1).clone()
            penalty = torch.cat((penalty[:, None], penalty[:, None], penalty[:, None]), -1).clone()
            obj = [obj, penalty]
            best_reward, best_index = ((obj[0][:, 0] + obj[1][:, 0]).view(batch_size, k)).min(-1)
        else:
            obj = torch.cat((obj[:, None], obj[:, None], obj[:, None]), -1).clone()
            best_reward, best_index = (obj[:, 0].view(batch_size, k)).min(-1)
        total_history = self.trainer_params["total_history"]
        if self.model_params["n2s_decoder"]:
            context2 = torch.zeros(batch_size * k, 4, solution_size)
            feasibility_history = torch.zeros(batch_size * k, total_history, solution_size)
        else:
            context2 = torch.zeros(batch_size * k, 9)
            context2[:, -1] = feasibility_history.view(-1) # current feasibility
            feasibility_history = feasibility_history.view(-1, 1).expand(batch_size * k, total_history)
        action = None
        # get the best solution from constrution, shape: (batch, solution_size)
        if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"] and self.trainer_params["select_strategy"] != "quality":
            # in this case, the selected solutions will not be the best
            best_reward, best_index = (-cons_reward.reshape(batch_size, env.pomo_size)).min(-1)
            best_solution = env.selected_node_list.reshape(batch_size, env.pomo_size, -1)[torch.arange(batch_size),best_index, :].clone()
            if "TSP" not in self.args.problem: best_solution = get_solution_with_dummy_depot(best_solution.view(batch_size, 1, -1), env.problem_size)
            rec_best = sol2rec(best_solution).view(batch_size, -1)
        else:
            rec_best = rec.view(batch_size, k, -1)[torch.arange(batch_size),best_index,:].clone()
        # print(f"!!!!!!!constructed best: {remove_dummy_depot_from_solution(rec2sol(rec_best).view(batch_size, 1, -1), env.problem_size)[:3]}")
        is_improved = torch.zeros(batch_size).bool()
        # log top k
        if self.args.problem == "VRPBLTW":
            self.metric_logger.improve_metrics["cons_tw_out_ratio"].update((out_penalty[:, 0] > 0).float().mean(), batch_size)
            self.metric_logger.improve_metrics["cons_capacity_out_ratio"].update((out_penalty[:, 1] > 0).float().mean(),batch_size)
            self.metric_logger.improve_metrics["cons_backhaul_out_ratio"].update((out_penalty[:, 2] > 0).float().mean(), batch_size)
            self.metric_logger.improve_metrics["cons_dlout_ratio"].update((out_penalty[:, 3] > 0).float().mean(), batch_size)
            self.metric_logger.improve_metrics["cons_out_ratio"].update((out_penalty>0).float().mean().item(), batch_size)

        # sample trajectory
        T = self.trainer_params["improve_steps"]
        T_pi = self.trainer_params["dummy_improve_steps"]
        '''
        logics for use the top T steps (improved reward) among T_pi steps to update the improvement model:
        1. with torch.no_grad(): conduct T_pi-steps improvements, record [the metrics w/o gradient] and the model input (rec, context, context2, action)
        Note: only record top T improved reward
        To achieve this: maintain a list of the index of top T steps and select them afterwards (remember to extend that to the batch_rewards)
        2. rerun the improvement step with top T model inputs, and get log p with gradient 
        '''
        use_LoRA = False
        memory = Memory()
        t = 0
        if self.trainer_params["shared_critic"]:
            bl_val_detached_list = []
            bl_val_list = []
            obj_of_impr = []
        if T_pi > 0:
            # step 1: rollout
            rec_list, context_list, context2_list, action_list, entropy = [], [], [], [], []
            with torch.no_grad():
                while t < T_pi:
                    state = (env, rec, context, context2, action)
                    rec_list.append(rec)
                    context_list.append(context)
                    context2_list.append(context2)
                    action_list.append(action)
                    if self.model_params["use_LoRA"] and t >= self.trainer_params["LoRA_begin_step"]: use_LoRA = True
                    with torch.amp.autocast(device_type="cuda", enabled=amp_training):
                        action, _, entro_p, improvement_method = self.model(state, solver="improvement", require_entropy=True, use_LoRA=use_LoRA)
                    entropy.append(entro_p)
                    # state transient
                    rec, rewards, obj, feasibility_history, context, context2, info, out_penalty, out_node_penalty = env.improvement_step(
                        rec, action, obj, feasibility_history, t,
                        improvement_method=improvement_method, epsilon=self.trainer_params["epsilon"],
                        seperate_obj_penalty=self.trainer_params["seperate_obj_penalty"],
                        weights=weights, out_reward=self.trainer_params["out_reward"],
                        penalty_factor=self.lambda_, penalty_normalize=self.trainer_params["penalty_normalize"],
                        insert_before=self.trainer_params["insert_before"],
                        non_linear=self.trainer_params["non_linear"], n2s_decoder=self.model_params["n2s_decoder"])
                    if self.trainer_params["bonus_for_construction"] or self.trainer_params["extra_bonus"] or self.trainer_params["imitation_learning"] or (self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]):
                        # update best solution
                        criterion = obj.clone() if not self.trainer_params["seperate_obj_penalty"] else (
                                    obj[0] + obj[1]).clone()
                        new_best, best_index = criterion[:, 0].view(batch_size, k).min(-1)
                        index = new_best < best_reward
                        best_reward[index] = new_best[index]  # update best reward
                        is_improved = (is_improved | index)
                        rec_best[index] = rec.view(batch_size, k, -1)[torch.arange(batch_size), best_index, :][index].clone()  # update best solution
                    memory.rewards.append(rewards)
                    criterion = obj.clone() if not self.trainer_params["seperate_obj_penalty"] else (obj[0] + obj[1]).clone()
                    memory.obj.append(criterion.clone())
                    memory.out_node_penalty.append(out_node_penalty)  # (c, b*k)
                    memory.out_penalty.append(out_penalty)  # (c, b*k)
                    feasible = out_penalty.sum(0) <= 0.0
                    soft_infeasible = (out_penalty.sum(0) <= epsilon) & (out_penalty.sum(0) > 0.)
                    memory.feasible.append(feasible)
                    memory.soft_feasible.append(soft_infeasible)
                    # next
                    t = t + 1
            # step 2: select the top T reward
            rollout_reward = torch.stack(memory.rewards)
            if self.trainer_params["dummy_improve_selected"] == "topk":
                reward, index_list = rollout_reward.sum(-1).topk(k=T, largest=True, dim=0) # shape: (T, batch)
            elif self.trainer_params["dummy_improve_selected"] == "random":
                index_list = torch.stack([torch.randperm(rollout_reward.size(0))[:T] for _ in range(rollout_reward.size(1))], dim=1)
                reward = rollout_reward.sum(-1)[index_list, torch.arange(rollout_reward.size(1))]  # shape: (T, batch)
            else:
                raise ValueError("Unknown dummy_improve_selected method: {}".format(self.trainer_params["dummy_improve_selected"]))
            if self.model.training: batch_reward.extend(list(torch.gather(rollout_reward[:, :, 0], 0, index_list).clone()))
            selected_rec = select_data_by_index(torch.stack(rec_list), index_list)
            selected_context2 = select_data_by_index(torch.stack(context2_list), index_list)
            action_list[0] = -torch.ones_like(action_list[-1])
            selected_action = select_data_by_index(torch.stack(action_list), index_list)
            context_list1 = tuple(torch.stack([data_tuple[i] for data_tuple in context_list], dim=0) for i in range(len(context_list[0])))
            selected_context = [select_data_by_index(data, index_list) for data in context_list1]
            selected_context = [tuple(tensors) for tensors in zip(*selected_context)]
            # step 3: rerun with gradient
            t = 0
            while t < T:
                state = (env, selected_rec[t], selected_context[t], selected_context2[t], selected_action[t])
                if self.model_params["use_LoRA"] and t >= self.trainer_params["LoRA_begin_step"]: use_LoRA=True
                with torch.amp.autocast(device_type="cuda", enabled=amp_training):
                    _, log_lh, _ = self.model(state, solver="improvement", require_entropy=False, use_LoRA=use_LoRA)
                if self.model.training: memory.logprobs.append(log_lh.clone())
                t += 1
        else:
            while t < T:
                # print(">>>>>>>>>> ", t)
                entropy = []

                state = (env, rec, context, context2, action)
                if self.model_params["use_LoRA"] and t >= self.trainer_params["LoRA_begin_step"]: use_LoRA=True
                with torch.amp.autocast(device_type="cuda", enabled=amp_training):
                    action, log_lh, entro_p, improvement_method, to_critic_ = self.model(state, solver="improvement", require_entropy=True, use_LoRA=use_LoRA)
                    if self.trainer_params["shared_critic"]:
                        baseline_val_detached, baseline_val = self.critic_improve(
                            to_critic_, obj[:, 1:2]
                        )
                        bl_val_detached_list.append(baseline_val_detached)
                        bl_val_list.append(baseline_val)
                        obj_of_impr.append(obj[:, 0])

                if self.model.training:
                    memory.logprobs.append(log_lh.clone())
                entropy.append(entro_p)

                # state transient
                rec, rewards, obj, feasibility_history, context, context2, info, out_penalty, out_node_penalty = env.improvement_step(rec, action, obj, feasibility_history, t,
                                                                                                 improvement_method = improvement_method, epsilon = self.trainer_params["epsilon"],
                                                                                                 seperate_obj_penalty=self.trainer_params["seperate_obj_penalty"],
                                                                                                 weights=weights, out_reward = self.trainer_params["out_reward"],
                                                                                                 penalty_factor=self.lambda_, penalty_normalize=self.trainer_params["penalty_normalize"],
                                                                                                 insert_before=self.trainer_params["insert_before"], non_linear=self.trainer_params["non_linear"],
                                                                                                 n2s_decoder=self.model_params["n2s_decoder"])

                if self.trainer_params["bonus_for_construction"] or self.trainer_params["extra_bonus"] or self.trainer_params["imitation_learning"] or (self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]):
                    # update best solution
                    criterion = obj.clone() if not self.trainer_params["seperate_obj_penalty"] else (obj[0]+obj[1]).clone()
                    new_best, best_index = criterion[:, 0].view(batch_size, k).min(-1)
                    index = new_best < best_reward
                    best_reward[index] = new_best[index] # update best reward
                    is_improved = (is_improved | index)
                    rec_best[index] = rec.view(batch_size, k, -1)[torch.arange(batch_size), best_index, :][index].clone() # update best solution

                if self.model.training: batch_reward.append(rewards[:, 0].clone())
                memory.rewards.append(rewards)
                criterion = obj.clone() if not self.trainer_params["seperate_obj_penalty"] else (obj[0] + obj[1]).clone()
                memory.obj.append(criterion.clone())
                # if self.args.problem == "CVRP":
                #     memory.cum_demand.append(context[2])
                #     memory.partial_sum_wrt_route_plan.append(context[3])
                #     non_feasible_cost_total = torch.clamp_min(context[-1] - 1.00001, 0.0).sum(-1)
                # elif self.args.problem == "TSPTW":
                #     exceed_time_window = torch.clamp_min(context[1] - context[-1], 0.0)
                #     if self.trainer_params["penalty_normalize"]:
                #         try:
                #             exceed_time_window = exceed_time_window / context[-1][:, 0]
                #         except:
                #             exceed_time_window = exceed_time_window / context[-1][:, :1]
                #     out_node_penalty = (exceed_time_window > 1e-5).sum(-1) # (b*k)
                #     memory.out_node_penalty.append(out_node_penalty)
                #     non_feasible_cost_total = exceed_time_window.sum(-1)
                #     memory.out_penalty.append(non_feasible_cost_total) # (b*k)
                memory.out_node_penalty.append(out_node_penalty)  # (c, b*k)
                memory.out_penalty.append(out_penalty)  # (c, b*k)
                feasible = out_penalty.sum(0) <= 0.0
                soft_infeasible = (out_penalty.sum(0) <= epsilon) & (out_penalty.sum(0) > 0.)
                memory.feasible.append(feasible)
                memory.soft_feasible.append(soft_infeasible)

                # next
                t = t + 1

        # calculate improvement loss
        if self.model.training:
            if self.trainer_params["shared_critic"]:
                bl_val_detached = torch.stack(bl_val_detached_list)
                bl_val = torch.stack(bl_val_list)
                # get td_traget value for critic
                Reward_list = []
                reward_reversed = memory.rewards[::-1]
                reward_reversed = [r / 1000. for r in reward_reversed]
                R = self.critic_improve(self.model((env, rec, context, context2, action), solver="improvement", only_critic=True),
                                        best_cost= obj[:, 1:2])[0]
                R = R / 1000.
                for r in range(len(reward_reversed)):
                    R = R * 0.999 + reward_reversed[r].sum(-1)
                    Reward_list.append(R)
                Reward = torch.stack(Reward_list[::-1], 0)
                # td_delta = td_target - critic(old_states)
                advantage = (Reward - bl_val_detached/1000.).detach()
                # print(Reward.mean(), bl_val_detached.mean(), advantage.mean())
                baseline_loss = (((bl_val/1000. - Reward)) ** 2)
                log_prob = torch.stack(memory.logprobs)
                loss = - advantage * log_prob  # Minus Sign: To Increase REWARD
                self.metric_logger.improve_metrics["actor_loss"].update(loss.mean().item(), batch_size)
                self.metric_logger.improve_metrics["critic_loss"].update(baseline_loss.mean().item(), batch_size)
                loss = loss + baseline_loss
                # calculate baseline for construction
                bl_construct_detach, bl_construct, trust_degree = self.critic_construct(obj_of_impr, bl_val_detached_list)
            elif self.trainer_params["baseline"] != "share":
                log_prob = torch.stack(memory.logprobs).view(T, batch_size, k)
                reward = torch.stack(memory.rewards).sum(-1).view(T, batch_size, k) if T_pi == 0 else reward.view(T, batch_size, k)
                baseline = 0. if self.trainer_params["baseline"] == "improve" else reward.mean(-1, keepdims=True)
                advantage = reward - baseline
                loss = - advantage * log_prob  # Minus Sign: To Increase REWARD
                # shape: (T, batch, pomo)
                # OLD IMPLEMENTATION
                # if self.model_params["use_LoRA"]:
                #     loss_mean = loss.mean(-1).mean(-1) # shape: (T,)
                #     self.metric_logger.improve_metrics["loss"].update(loss.mean().item(), batch_size)
                #     self.metric_logger.improve_metrics["large_model_loss"].update(loss[:self.trainer_params["LoRA_begin_step"]].mean().item(), batch_size)
                #     self.metric_logger.improve_metrics["lora_loss"].update(loss[self.trainer_params["LoRA_begin_step"]:].mean().item(), batch_size)
            else:
                impr_log_prob = torch.stack(memory.logprobs).view(T, batch_size, k).sum(0)  # shape: (batch, k)
                cons_log_prob = torch.gather(cons_log_prob, 1, select_idx)  # shape: (batch, k)
                log_prob = impr_log_prob + cons_log_prob  # (log Pc).sum() + (log Pi).sum() = (log Pc*Pi).sum()
                # first min: min during T steps
                score = torch.stack(memory.obj)  # shape: (T, batch*pomo)
                min_cons_n_impr = score.min(dim=0)[0].view(batch_size, k, -1)[:, :, 1]  # shape: (batch, k)
                advantage = min_cons_n_impr - min_cons_n_impr.float().mean(dim=1, keepdims=True)  # shape: (batch, k)
                loss = - advantage * log_prob  # Minus Sign: To Increase REWARD # shape: (batch, k)
            loss_mean = loss.mean()
            self.metric_logger.improve_metrics["loss"].update(loss_mean.item(), batch_size)

        # entropy
        entropy = torch.stack(entropy).mean().item()
        self.metric_logger.improve_metrics["entropy"].update(entropy, batch_size)

        # score = cost + penalty
        score = torch.stack(memory.obj) # shape: (T, batch*pomo)
        if (self.trainer_params["bonus_for_construction"] or self.trainer_params["extra_bonus"]) and self.trainer_params["baseline"] != "improve":
            improve_reward = score[:, :, 0].min(dim=0)[0].view(batch_size, k) # output the bsf during improvement for every initial solutions
        else:
            improve_reward = score.min(dim=0)[0].view(batch_size, k, -1).min(dim=1)[0][:, 0]
            # shape: (batch)  # ATTENTION: WHY USING .min(dim=1)[0] AS THE IMPROVE BASELINE, NOT MEAN?
        score_mean = score.min(dim=0)[0].view(batch_size, k, -1).min(dim=1)[0].mean(0) # 3, i.e., (current, bsf, tsp_bsf)
        self.metric_logger.improve_metrics["current_score"].update(score_mean[0].item(), batch_size) # improve
        self.metric_logger.improve_metrics["bsf_score"].update(score_mean[1].item(), batch_size) # improve + construct
        self.metric_logger.improve_metrics["epsilon_fsb_bsf_score"].update(score_mean[2].item(), batch_size)

        # reward
        reward_mean = torch.stack(memory.rewards) # (T, batch_size, k, 3) 3 for imrpove_reward, reg to jump, bonus to explore e-feasible
        self.metric_logger.improve_metrics["improve_reward"].update(reward_mean[:, :, 0].mean().item(), batch_size)  # improve_reward
        self.metric_logger.improve_metrics["reg_reward"].update(reward_mean[:, :, 1].mean().item(), batch_size)  # reg to jump
        self.metric_logger.improve_metrics["bonus_reward"].update(reward_mean[:, :, 2].mean().item(), batch_size)  # bonus to explore e-feasible

        # penalty
        # if self.args.problem == "CVRP":
        #     out_penalty = torch.stack(memory.partial_sum_wrt_route_plan) # shape: (T, batch*pomo, solution)
        #     out_penalty = ((out_penalty - 1.00001) * (out_penalty > 1.00001)).sum(dim=-1).min(dim=0)[0].view(batch_size, k)
        #     out_node_penalty = torch.stack(memory.cum_demand) # shape: (T, batch*pomo, solution)
        #     out_node_penalty = (out_node_penalty > 1.00001).sum(-1).min(dim=0)[0].view(batch_size, k)
        # elif self.args.problem == "TSPTW":
        #     out_penalty = torch.stack(memory.out_penalty)  # shape: (T, batch*pomo)
        #     out_node_penalty = torch.stack(memory.out_node_penalty)  # shape: (T, batch*pomo)
        out_penalty = torch.stack(memory.out_penalty)  # shape: (T, (c), batch*pomo)
        out_node_penalty = torch.stack(memory.out_node_penalty)  # shape: (T, (c), batch*pomo)
        if self.args.problem == "VRPBLTW":
            min_out_penalty = out_penalty[:,0].min(dim=1)[0].float().mean()  # get best results from pomo
            min_out_node_penalty = out_node_penalty[:,0].min(dim=1)[0].float().mean()  # get best results from pomo
            self.metric_logger.improve_metrics["tw_out"].update(out_penalty[:,0].mean().item(), batch_size)
            self.metric_logger.improve_metrics["tw_out_nodes"].update(out_node_penalty[:,0].float().mean().item(), batch_size)
            self.metric_logger.improve_metrics["tw_out_ratio"].update((out_penalty[:,0] > 0).float().mean(), batch_size)
            # self.metric_logger.improve_metrics["tw_out"].update(min_out_penalty.item(), batch_size)
            # self.metric_logger.improve_metrics["tw_out_nodes"].update(min_out_node_penalty.item(), batch_size)
            min_out_penalty = out_penalty[:,1].min(dim=1)[0].float().mean()  # get best results from pomo
            min_out_node_penalty = out_node_penalty[:,1].min(dim=1)[0].float().mean()  # get best results from pomo
            self.metric_logger.improve_metrics["capacity_out"].update(out_penalty[:,1].mean().item(), batch_size)
            self.metric_logger.improve_metrics["capacity_out_nodes"].update(out_node_penalty[:,1].float().mean().item(), batch_size)
            self.metric_logger.improve_metrics["capacity_out_ratio"].update((out_penalty[:, 1] > 0).float().mean(), batch_size)
            # self.metric_logger.improve_metrics["capacity_out"].update(min_out_penalty.item(), batch_size)
            # self.metric_logger.improve_metrics["capacity_out_nodes"].update(min_out_node_penalty.item(), batch_size)
            min_out_penalty = out_penalty[:,2].min(dim=1)[0].float().mean()  # get best results from pomo
            min_out_node_penalty = out_node_penalty[:, 2].min(dim=1)[0].float().mean()  # get best results from pomo
            self.metric_logger.improve_metrics["backhaul_out"].update(out_penalty[:,2].mean().item(), batch_size)
            self.metric_logger.improve_metrics["backhaul_out_nodes"].update(out_node_penalty[:,2].float().mean().item(), batch_size)
            self.metric_logger.improve_metrics["backhaul_out_ratio"].update((out_penalty[:, 2] > 0).float().mean(), batch_size)
            # self.metric_logger.improve_metrics["backhaul_out"].update(min_out_penalty.item(), batch_size)
            # self.metric_logger.improve_metrics["backhaul_out_nodes"].update(min_out_node_penalty.item(), batch_size)
            min_out_penalty = out_penalty[:,3].min(dim=1)[0].float().mean()  # get best results from pomo
            min_out_node_penalty = out_node_penalty[:,3].min(dim=1)[0].float().mean()  # get best results from pomo
            self.metric_logger.improve_metrics["dlout"].update(out_penalty[:,3].mean().item(), batch_size)
            self.metric_logger.improve_metrics["dlout_nodes"].update(out_node_penalty[:,3].float().mean().item(), batch_size)
            self.metric_logger.improve_metrics["dlout_ratio"].update((out_penalty[:,3] > 0).float().mean(), batch_size)
            # self.metric_logger.improve_metrics["dlout"].update(min_out_penalty.item(), batch_size)
            # self.metric_logger.improve_metrics["dlout_nodes"].update(min_out_node_penalty.item(), batch_size)
            out_penalty = out_penalty.sum(1)
            out_node_penalty = out_node_penalty.sum(1)
        min_out_penalty = out_penalty.min(dim=1)[0].float().mean()  # get best results from pomo
        min_out_node_penalty = out_node_penalty.min(dim=1)[0].float().mean()  # get best results from pomo
        # get best from T steps
        self.metric_logger.improve_metrics["out"].update(out_penalty.float().min(0)[0].mean().item(), batch_size)
        self.metric_logger.improve_metrics["out_nodes"].update(out_node_penalty.float().min(0)[0].mean().item(), batch_size)
        self.metric_logger.improve_metrics["out_ratio"].update((out_penalty>0).min(0)[0].float().mean().item(), batch_size)
        # self.metric_logger.improve_metrics["out"].update(min_out_penalty.item(), batch_size)
        # self.metric_logger.improve_metrics["out_nodes"].update(min_out_node_penalty.item(), batch_size)

        # calculate infeasible outputs (BSF during improvement)
        feasible_all = torch.stack(memory.feasible) # shape: (T, batch*pomo)
        soft_feasible_all = torch.stack(memory.soft_feasible)# shape: (T, batch*pomo)
        feasible_during_impr = feasible_all.any(0)# shape: (batch*pomo)
        batch_feasible = feasible_during_impr.view(batch_size, k).any(-1)# shape: (batch)
        soft_feasible_during_impr = soft_feasible_all.any(0)# shape: (batch*pomo)
        soft_batch_feasible = soft_feasible_during_impr.view(batch_size, k).any(-1)# shape: (batch)
        # infeasible = ~(feasible_all.any(0))# shape: (batch*pomo)
        # batch_feasible = (infeasible.view(batch_size, k)==False).any(dim=-1) # shape: (batch)
        # soft_infeasible = ~(soft_feasible_all.any(0))# shape: (batch*pomo)
        # soft_batch_feasible = (soft_infeasible.view(batch_size, k) == False).any(dim=-1)  # shape: (batch)
        # self.metric_logger.improve_metrics["sol_infeasible_rate"].update(infeasible.sum() / (batch_size * k), batch_size)
        self.metric_logger.improve_metrics["sol_infeasible_rate"].update(1. - feasible_during_impr.sum() / (batch_size * k), batch_size * k)
        self.metric_logger.improve_metrics["ins_infeasible_rate"].update(1. - batch_feasible.sum() / batch_size, batch_size)
        # self.metric_logger.improve_metrics["soft_sol_infeasible_rate"].update(soft_infeasible.sum() / (batch_size * k), batch_size)
        self.metric_logger.improve_metrics["soft_sol_infeasible_rate"].update(1. - soft_feasible_during_impr.sum() / (batch_size * k), batch_size *k)
        self.metric_logger.improve_metrics["soft_ins_infeasible_rate"].update(1. - soft_batch_feasible.sum() / batch_size, batch_size)

        if feasible_all.any(): # BSF during improvement
            score_bsf = score[:, :, 0].clone().masked_fill(~feasible_all, 1e10).min(0)[0]
            feasible_dist = torch.where(~feasible_all.any(0), torch.zeros_like(score_bsf), score_bsf) # feasible dist left only
            feasible_dist_mean = feasible_dist.sum() / feasible_all.any(0).sum()
            feasible_dist_max_pomo = score_bsf.view(batch_size, k).min(dim=1)[0][batch_feasible].sum() / batch_feasible.sum() # get best results from pomo, shape: (batch)[batch_feasible]
            self.metric_logger.improve_metrics["feasible_dist_mean"].update(feasible_dist_mean, feasible_all.any(0).sum())
            self.metric_logger.improve_metrics["feasible_dist_max_pomo_mean"].update(feasible_dist_max_pomo, batch_feasible.sum())

        if soft_feasible_all.any(): # BSF during improvement
            soft_score_bsf = score[:, :, 0].clone().masked_fill(~soft_feasible_all, 1e10).min(0)[0]
            epision_feasible_dist = torch.where(~soft_feasible_all.any(0), torch.zeros_like(soft_score_bsf), soft_score_bsf) # feasible dist left only
            epision_feasible_dist_mean = epision_feasible_dist.sum() / soft_feasible_all.any(0).sum()  # calculate mean
            epision_feasible_max_pomo_dist = soft_score_bsf.view(batch_size, k).min(dim=1)[0][soft_batch_feasible].sum()  # get best results from pomo, shape: (batch)
            self.metric_logger.improve_metrics["epsilon_feasible_dist_mean"].update(epision_feasible_dist_mean, soft_feasible_all.any(0).sum())
            self.metric_logger.improve_metrics["epsilon_feasible_dist_max_pomo_mean"].update(epision_feasible_max_pomo_dist, soft_batch_feasible.sum())

        # just to test
        # topk = torch.stack(memory.obj)[:, :, 1].min(0)[0].view(batch_size, 40).topk(k=5, dim=-1, largest=False).indices
        # overlap = row_wise_overlap_no_loop(topk, topk2)
        # print(overlap, overlap.float().mean().item())

        # end update
        memory.clear_memory()
        # print(f"!!!!!!!improved best: {remove_dummy_depot_from_solution(rec2sol(rec_best).view(batch_size, 1, -1), env.problem_size)[:3]}")

        return loss_mean, improve_reward, select_idx, remove_dummy_depot_from_solution(rec2sol(rec_best).view(batch_size, 1, -1), env.problem_size), best_reward, is_improved, \
                bl_construct_detach if self.trainer_params["shared_critic"] else None, bl_construct if self.trainer_params["shared_critic"] else None

    def _val_improvement(self, env, aug_factor, cons_reward, best_solution=None):

        with torch.no_grad():
            # solution/rec shape: (batch, pomo, solution)
            solution = env.selected_node_list.clone()

            if not self.trainer_params["improvement_only"]:  # if yes, already generate k solutions for each instance
                # select top k
                # if self.trainer_params["select_top_k_val"] == 1 and best_solution is not None:
                #     batch_size = best_solution.size(0)
                #     solution = best_solution.clone().view(batch_size, 1, -1)
                #     feasibility_history = self.val_metric_logger.best_feasible[-batch_size:].clone().view(batch_size, 1)
                # else:
                # FIXME: why not select the best in the feasible ones then the most feasible ones among the infeasible
                if self.trainer_params["select_top_k_val"] <= env.pomo_size:
                    solution, select_idx = select4improve(solution, cons_reward,
                                                          strategy=self.trainer_params["select_strategy"],
                                                          K=self.trainer_params["select_top_k_val"],
                                                          rnd_prob=self.trainer_params["stochastic_probability"],
                                                          diversity=self.trainer_params["diversity"])
                    # solution shape: (batch, k, solution); solution_idx shape: (batch, k)
                    feasibility_history = torch.gather(~env.infeasible, 1, select_idx)
                    if self.trainer_params["neighborhood_search"]:
                        cons_log_prob = cons_log_prob[torch.arange(env.batch_size)[:, None], select_idx]
                        _, unconfident_indices = torch.topk(cons_log_prob, k=self.trainer_params["k_unconfident"], dim=-1, largest=False)
                else:
                    # just for testing
                    select_idx = torch.arange(env.pomo_size)[None, :].repeat(env.batch_size, 1)
                    feasibility_history = ~env.infeasible
                    _, topk2 = select4improve(solution, cons_reward, strategy=self.trainer_params["select_strategy"],
                                              K=5, rnd_prob=self.trainer_params["stochastic_probability"],
                                              diversity=self.trainer_params["diversity"])
                feasibility_history_clone = feasibility_history.clone()
            else:
                select_idx = torch.arange(env.pomo_size)[None, :].repeat(env.batch_size, 1)
                feasibility_history = ~env.infeasible
                feasibility_history_clone = feasibility_history.clone()


            if "TSP" not in self.args.problem: solution = get_solution_with_dummy_depot(solution, env.problem_size)
            batch_size, pomo_size, solution_size = solution.size() # batch_size = aug_factor * batch_size
            print(solution.size())
            rec = sol2rec(solution).view(batch_size * pomo_size, -1)

            # preapare input
            obj, context, out_penalty, out_node_penalty = env.get_costs(rec, get_context=True) # obj only
            obj = torch.cat((obj[:, None], obj[:, None], obj[:, None]), -1).clone()
            # obj = obj.unsqueeze(-1).expand(-1, -1, 3)
            total_history = self.trainer_params["total_history"]
            if self.model_params["n2s_decoder"]:
                context2 = torch.zeros(batch_size * pomo_size, 4, solution_size)
                feasibility_history = torch.zeros(batch_size * pomo_size, total_history, solution_size)
            else:
                context2 = torch.zeros(batch_size * pomo_size, 9)
                context2[:, -1] = (feasibility_history).view(-1) # current feasibility
                feasibility_history = (feasibility_history).view(-1, 1).expand(batch_size * pomo_size, total_history)
            action = None
            best_reward, best_index = (obj[:, 0].view(batch_size//aug_factor, aug_factor*pomo_size)).min(-1)
            rec_best = rec.view(batch_size//aug_factor, aug_factor*pomo_size, -1)[torch.arange(batch_size//aug_factor), best_index, :].clone()
            # print(f"!!!!!!!constructed best: {remove_dummy_depot_from_solution(rec2sol(rec_best).view(batch_size, 1, -1), env.problem_size)[:3]}")
            is_improved = torch.zeros(batch_size//aug_factor).bool()
            if self.tester_params["best_solution_path"] is not None:
                tmp_obj = reshape_aug_view(obj[:, 0].clone(), batch_size, aug_factor, pomo_size)
                tmp_solution = reshape_aug_solution(remove_dummy_depot_from_solution(rec2sol(rec).unsqueeze(1), env.problem_size), batch_size, aug_factor, pomo_size).clone()
                tmp_penalty = reshape_aug_view(out_penalty.sum(0) + out_node_penalty.sum(0), batch_size, aug_factor, pomo_size).clone()
                tmp_feasible = reshape_aug_view((out_penalty.sum(0) + out_node_penalty.sum(0)) < 1e-5, batch_size, aug_factor, pomo_size).clone()
                new_best_reward, new_feasible, new_best_solution = cal_best_aug_batch(
                    dist=tmp_obj.T,
                    solution=tmp_solution.transpose(0, 1),
                    penalty=tmp_penalty.T,
                    feasible=tmp_feasible.T
                )

                update_best_records(
                        best_reward=self.val_metric_logger.best_reward,
                        best_feasible=self.val_metric_logger.best_feasible,
                        best_solution=self.val_metric_logger.best_solution,
                        new_reward = new_best_reward,
                        new_feasible = new_feasible,
                        new_solution = new_best_solution
                )

                self.val_metric_logger.refinement_reward_bsf_history = torch.cat([self.val_metric_logger.refinement_reward_bsf_history, self.val_metric_logger.best_reward.unsqueeze(-1)], dim=-1)
                self.val_metric_logger.refinement_feasible_bsf_history = torch.cat([self.val_metric_logger.refinement_feasible_bsf_history, self.val_metric_logger.best_feasible.unsqueeze(-1)], dim=-1)
                self.val_metric_logger.refinement_solution_history = remove_dummy_depot_from_solution(rec2sol(rec).unsqueeze(1), env.problem_size).unsqueeze(1) #batch, 0~T, solution
            # sample trajectory
            t = 0
            T = self.trainer_params["validation_improve_steps"]
            use_LoRA = False
            # memory = Memory()
            # initial solution from construction
            feasible_all = ((feasibility_history_clone).view(-1)).int()
            min_scores = torch.full((batch_size * pomo_size,), float('inf'))
            min_scores = torch.where(feasible_all.bool(), obj[:, 0], min_scores).clone()
            out_node_penalties = torch.full((batch_size * pomo_size,), float('inf'))
            # if self.args.problem == "CVRP":
            #     out_node_penalty = (context[2] > 1.00001).sum(-1)
            # elif self.args.problem == "TSPTW":
            #     out_node_penalty = (torch.clamp_min(context[1] - context[-1], 0.0) > 1e-5).sum(-1) # (b*k)
            out_node_penalties = torch.where(~(feasible_all.bool()), out_node_penalty, out_node_penalties).clone()
            del out_node_penalty
            out_penalties = torch.full((batch_size * pomo_size,), float('inf'))
            # if self.args.problem == "CVRP":
            #     out_penalty = ((context[3] - 1.00001) * (context[3] > 1.00001)).sum(dim=1)
            # elif self.args.problem == "TSPTW":
            #     out_penalty = torch.clamp_min(context[1] - context[-1], 0.0).sum(-1)
            #     if self.trainer_params["penalty_normalize"]:
            #         try:
            #             out_penalty = out_penalty / context[-1][:, 0]
            #         except:
            #             out_penalty = out_penalty / context[-1][:, :1]
            out_penalties = torch.where(~(feasible_all.bool()), out_penalty, out_penalties).clone()
            del out_penalty

            while t < T:
                # print(t)

                state = (env, rec, context, context2, action)

                if self.model_params["use_LoRA"] and t >= self.trainer_params["LoRA_begin_step"]: use_LoRA = True
                action, _, improvement_method = self.model(state, solver="improvement", require_entropy=False, use_LoRA=use_LoRA)

                # state transient
                # rec, rewards, obj, feasibility_history, context, context2, info
                rec, _, obj, feasibility_history, context, context2, _, out_penalty, out_node_penalty = env.improvement_step(rec, action, obj, feasibility_history, t,
                                                                                                                       improvement_method = improvement_method, insert_before=self.trainer_params["insert_before"], n2s_decoder=self.model_params["n2s_decoder"])
                # update best solution
                tmp_obj = obj[:, 0].clone().reshape(aug_factor, batch_size // aug_factor, pomo_size).permute(1, 0, 2).reshape(batch_size // aug_factor, aug_factor * pomo_size)
                best_reward, best_index = (tmp_obj).min(-1)
                new_best, best_index = tmp_obj.min(-1)
                index = new_best < best_reward
                best_reward[index] = new_best[index]  # update best reward
                is_improved = (is_improved | index)
                tmp_rec = rec.reshape(aug_factor, batch_size // aug_factor, pomo_size, -1).permute(1, 0, 2, 3).reshape(batch_size // aug_factor, aug_factor * pomo_size, -1)
                rec_best[index] = tmp_rec[torch.arange(batch_size//aug_factor), best_index, :][index].clone()  # update best solution
                if self.tester_params["best_solution_path"] is not None:
                    tmp_obj = reshape_aug_view(obj[:, 0].clone(), batch_size, aug_factor, pomo_size)
                    tmp_solution = reshape_aug_solution(remove_dummy_depot_from_solution(rec2sol(rec).unsqueeze(1), env.problem_size), batch_size, aug_factor,pomo_size).clone()
                    tmp_penalty = reshape_aug_view(out_penalty.sum(0) + out_node_penalty.sum(0), batch_size, aug_factor,pomo_size).clone()
                    tmp_feasible = reshape_aug_view((out_penalty.sum(0) + out_node_penalty.sum(0)) < 1e-5, batch_size, aug_factor,pomo_size).clone()
                    new_best_reward, new_feasible, new_best_solution = cal_best_aug_batch(
                        dist=tmp_obj.T,
                        solution=tmp_solution.transpose(0, 1),
                        penalty=tmp_penalty.T,
                        feasible=tmp_feasible.T
                    )

                    update_best_records(
                        best_reward=self.val_metric_logger.best_reward,
                        best_feasible=self.val_metric_logger.best_feasible,
                        best_solution=self.val_metric_logger.best_solution,
                        new_reward=new_best_reward,
                        new_feasible=new_feasible,
                        new_solution=new_best_solution
                    )
                    self.val_metric_logger.refinement_reward_bsf_history = torch.cat(
                        [self.val_metric_logger.refinement_reward_bsf_history,
                         self.val_metric_logger.best_reward.unsqueeze(-1)], dim=-1)
                    self.val_metric_logger.refinement_feasible_bsf_history = torch.cat(
                        [self.val_metric_logger.refinement_feasible_bsf_history,
                         self.val_metric_logger.best_feasible.unsqueeze(-1)], dim=-1)
                    self.val_metric_logger.refinement_solution_history = torch.cat([
                        self.val_metric_logger.refinement_solution_history,
                         remove_dummy_depot_from_solution(rec2sol(rec).unsqueeze(1), env.problem_size).unsqueeze(1)], dim=1
                    )# batch, T, solution
                # memory.obj.append(obj.clone())
                # memory.cum_demand.append(context[2])
                # memory.partial_sum_wrt_route_plan.append(context[3])
                # non_feasible_cost_total = torch.clamp_min(context[-1] - 1.00001, 0.0).sum(-1)
                # feasible = non_feasible_cost_total <= 0.0
                # memory.feasible.append(feasible)
                # if self.args.problem == "CVRP":
                #     non_feasible_cost_total = torch.clamp_min(context[-1] - 1.00001, 0.0).sum(-1)
                # elif self.args.problem == "TSPTW":
                #     non_feasible_cost_total = torch.clamp_min(context[1] - context[-1], 0.0).sum(-1)
                #     if self.trainer_params["penalty_normalize"]:
                #         try:
                #             non_feasible_cost_total = non_feasible_cost_total / context[-1][:, 0]
                #         except:
                #             non_feasible_cost_total = non_feasible_cost_total / context[-1][:, :1]
                feasible = out_penalty.sum(0) <= 0.0
                feasible_all += feasible.int()
                # Update scores with current step's results
                min_scores = torch.where(feasible, torch.min(min_scores, obj[:, 0]), min_scores).clone()

                # Update out_node_penalties and out_penalties
                # if self.args.problem == "CVRP":
                #     current_out_node_penalty = (context[2] > 1.00001).sum(-1)
                #     # current_out_penalty = (context[3] - 1.00001).clamp(min=0).sum(dim=-1)
                #     current_out_penalty = ((context[3] - 1.00001) * (context[3] > 1.00001)).sum(dim=1)
                # elif self.args.problem == "TSPTW":  # arrival_time - tw_end
                #     current_out_penalty = torch.clamp_min(context[1] - context[-1], 0.0).sum(-1)
                #     if self.trainer_params["penalty_normalize"]:
                #         try:
                #             current_out_penalty = current_out_penalty / context[-1][:, 0]
                #         except:
                #             current_out_penalty = current_out_penalty / context[-1][:, :1]
                #     current_out_node_penalty = (torch.clamp_min(context[1] - context[-1], 0.0) > 1e-5).sum(-1)  # (b*k)
                out_node_penalties = torch.min(out_node_penalties, out_node_penalty.sum(0)).clone()
                out_penalties = torch.min(out_penalties, out_penalty.sum(0)).clone()

                # next
                t = t + 1

            # calculate infeasible outputs (BSF during improvement)
            # feasible_all = torch.stack(memory.feasible) # shape: (T, aug*batch*pomo)
            # feasible = feasible_all.any(0)# shape: (aug*batch*pomo)
            feasible = feasible_all > 0 # shape: (aug*batch*pomo)
            aug_feasible = feasible.reshape(aug_factor, -1, pomo_size).any(dim=0).any(dim=-1) # shape: (batch,)
            no_aug_feasible = feasible.reshape(aug_factor, -1, pomo_size)[0].any(dim=-1) # shape: (batch,)
            self.val_metric_logger._improve_tensor_update("aug_feasible", aug_feasible)
            self.val_metric_logger._improve_tensor_update("no_aug_feasible", no_aug_feasible)
            sol_infeasible_rate = 1. - (feasible.sum() / (batch_size * pomo_size)) # batch_size = aug*batch
            ins_infeasible_rate = 1. - aug_feasible.sum() / (batch_size//aug_factor)
            self.val_metric_logger.improve_metrics["sol_infeasible_rate"].update(sol_infeasible_rate, batch_size * pomo_size)
            self.val_metric_logger.improve_metrics["ins_infeasible_rate"].update(ins_infeasible_rate, batch_size // aug_factor)

            # score = cost
            # score = torch.stack(memory.obj)[:, :, 0] # after each improvement step, shape: (T, aug*batch*pomo)
            # score_fsb = torch.where(~feasible_all, 1e10, score) # make the infeasible one to be a very large value
            # score_fsb = score_fsb.min(dim=0)[0].reshape(aug_factor, -1, pomo_size) # best during improvement
            # aug_score_fsb = score_fsb.min(dim=-1)[0].min(dim=0)[0] # shape: (batch,)
            # no_aug_score_fsb = score_fsb.min(dim=-1)[0][0]  # shape: (batch,)
            min_scores = min_scores.view(aug_factor, -1, pomo_size)
            aug_score_fsb = min_scores.min(dim=-1)[0].min(dim=0)[0]
            no_aug_score_fsb = min_scores[0].min(dim=-1)[0]
            self.val_metric_logger._improve_tensor_update("aug_score", aug_score_fsb)
            self.val_metric_logger._improve_tensor_update("no_aug_score", no_aug_score_fsb)

            # penalty
            # out_node_penalty = (torch.stack(memory.cum_demand)> 1.00001).sum(dim=-1) # (T, aug*batch*pomo)
            # out_node_penalty = out_node_penalty.min(dim=0)[0].reshape(aug_factor, -1, pomo_size)# best during improvement, shape: ( aug,batch,pomo)
            # no_aug_out_nodes = out_node_penalty[0].min(dim=-1)[0] # shape: (batch,)
            # aug_out_nodes = out_node_penalty.min(dim=0)[0].min(dim=-1)[0] # shape: (batch,)
            # partial_sum_wrt_route_plan = torch.stack(memory.partial_sum_wrt_route_plan)
            # out_penalty = ((partial_sum_wrt_route_plan - 1.00001) * (partial_sum_wrt_route_plan > 1.00001)).sum(dim=-1) # (T, aug*batch*pomo)
            # out_penalty = out_penalty.min(dim=0)[0].reshape(aug_factor, -1, pomo_size)  # best during improvement, shape: (aug,batch,pomo)
            # no_aug_out = out_penalty[0].min(dim=-1)[0] # shape: (batch,)
            # aug_out = out_penalty.min(dim=0)[0].min(dim=-1)[0] # shape: (batch,)
            out_node_penalties = out_node_penalties.view(aug_factor, -1, pomo_size)
            no_aug_out_nodes = out_node_penalties[0].min(dim=-1)[0]
            aug_out_nodes = out_node_penalties.min(dim=0)[0].min(dim=-1)[0]
            out_penalties = out_penalties.view(aug_factor, -1, pomo_size)
            no_aug_out = out_penalties[0].min(dim=-1)[0]
            aug_out = out_penalties.min(dim=0)[0].min(dim=-1)[0]
            self.val_metric_logger._improve_tensor_update("no_aug_out", no_aug_out)
            self.val_metric_logger._improve_tensor_update("no_aug_out_nodes", no_aug_out_nodes)
            self.val_metric_logger._improve_tensor_update("aug_out", aug_out)
            self.val_metric_logger._improve_tensor_update("aug_out_nodes", aug_out_nodes)

            # end update
            # memory.clear_memory()

            return aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible, remove_dummy_depot_from_solution(rec2sol(rec_best).view(batch_size//aug_factor, 1, -1), env.problem_size), is_improved

    def _val_improvement_with_eas(self, env, aug_factor, cons_reward, epsilon=None, batch_reward = None, weights = 0):
        assert self.trainer_params["select_top_k_val"] == 1, "Not implemented"

        # solution/rec shape: (batch, pomo, solution)
        solution = env.selected_node_list.clone()
        iterations = self.tester_params['EAS_params']['iterations_impr']
        enable_EAS = self.tester_params['EAS_params']['enable']
        self.model.kopt_decoder.reset_EAS_layers(solution.size(0))  # initialize/reset EAS layers
        EAS_layer_parameters = self.model.kopt_decoder.get_EAS_parameters()
        # Only store gradients for new EAS layer weights
        self.model.requires_grad_(False)
        for t in EAS_layer_parameters:
            t.requires_grad_(True)
        optimizer = Optimizer(EAS_layer_parameters, lr=self.tester_params['EAS_params']['lr'])
        self.model.train()

        if not self.trainer_params["improvement_only"]:  # if yes, already generate k solutions for each instance
            # select top k
            if self.trainer_params["select_top_k_val"] <= env.pomo_size:
                solution, select_idx = select4improve(solution, cons_reward,
                                                      strategy=self.trainer_params["select_strategy"],
                                                      K=self.trainer_params["select_top_k_val"],
                                                      rnd_prob=self.trainer_params["stochastic_probability"],
                                                      diversity=self.trainer_params["diversity"])
                # solution shape: (batch, k, solution); solution_idx shape: (batch, k)
                feasibility_history = torch.gather(~env.infeasible, 1, select_idx)
                if self.trainer_params["neighborhood_search"]:
                    cons_log_prob = cons_log_prob[torch.arange(env.batch_size)[:, None], select_idx]
                    _, unconfident_indices = torch.topk(cons_log_prob, k=self.trainer_params["k_unconfident"],
                                                        dim=-1, largest=False)
            else:
                # just for testing
                select_idx = torch.arange(env.pomo_size)[None, :].repeat(env.batch_size, 1)
                feasibility_history = ~env.infeasible
                _, topk2 = select4improve(solution, cons_reward, strategy=self.trainer_params["select_strategy"],
                                          K=5, rnd_prob=self.trainer_params["stochastic_probability"],
                                          diversity=self.trainer_params["diversity"])
            feasibility_history_clone = feasibility_history.clone()
        else:
            select_idx = torch.arange(env.pomo_size)[None, :].repeat(env.batch_size, 1)
            feasibility_history = ~env.infeasible
        total_history = self.trainer_params["total_history"]

        if batch_reward is None:
            batch_reward = []
            weights = 0
        else:
            try:
                batch_reward = torch.cat(batch_reward)
                if self.args.multiple_gpu:
                    dist.barrier()
                    batch_reward = gather_tensor_and_concat(batch_reward.contiguous())
                    dist.barrier()
                weights = batch_reward.mean()
                batch_reward = []
            except:
                batch_reward = []
                weights = 0
        print(solution.size())
        is_improved = torch.zeros(solution.size(0) // aug_factor).bool()

        for iter in tqdm(range(iterations)):

            # preapare input for each iter (the best solution from previous improvement)
            if "TSP" not in self.args.problem: solution = get_solution_with_dummy_depot(solution, env.problem_size)
            batch_size, pomo_size, solution_size = solution.size() # batch_size = aug_factor * batch_size
            rec = sol2rec(solution).view(batch_size * pomo_size, -1)
            obj, context, out_penalty, out_node_penalty = env.get_costs(rec, get_context=True, out_reward=self.trainer_params["out_reward"], penalty_factor = self.lambda_, seperate_obj_penalty=self.trainer_params["seperate_obj_penalty"])
            obj = torch.cat((obj[:, None], obj[:, None], obj[:, None]), -1).clone()
            score = obj.clone()
            feasibility_history = out_penalty.sum(0) <= 0.0
            if iter == 0: feasible_best = feasibility_history.clone()
            feasible_all = feasibility_history.clone()
            if self.model_params["n2s_decoder"]:
                context2 = torch.zeros(batch_size * pomo_size, 4, solution_size)
                feasibility_history = torch.zeros(batch_size * pomo_size, total_history, solution_size)
            else:
                context2 = torch.zeros(batch_size * pomo_size, 9)
                context2[:, -1] = (feasibility_history).view(-1) # current feasibility
                feasibility_history = (feasibility_history).view(-1, 1).expand(batch_size * pomo_size, total_history)
            action = None
            # best_reward, best_index = (obj[:, 0].view(batch_size//aug_factor, aug_factor*pomo_size)).min(-1)
            # rec_best = rec.view(batch_size//aug_factor, aug_factor*pomo_size, -1)[torch.arange(batch_size//aug_factor), best_index, :].clone()
            tmp_obj = obj[:, 0].clone().view(batch_size//aug_factor, aug_factor*pomo_size)
            tmp_fsb = feasible_all.clone().view(batch_size//aug_factor, aug_factor*pomo_size)
            best_reward, best_index = tmp_obj.masked_fill(~tmp_fsb, float("inf")).min(-1)
            no_feasible = torch.isinf(best_reward)
            if no_feasible.any():
                fallback_best, fallback_index = tmp_obj[no_feasible, :].min(dim=-1)
                best_reward[no_feasible] = fallback_best
                best_index[no_feasible] = fallback_index
            rec_best = rec.view(batch_size // aug_factor, aug_factor * pomo_size, -1)[torch.arange(batch_size // aug_factor), best_index, :].clone()
            rec_iter_history = rec.clone().unsqueeze(1)
            best_reward_iter = score[:,0].clone()
            rec_best_iter = rec.clone()

            # sample trajectory
            t = 0
            T = self.trainer_params["validation_improve_steps"]
            use_LoRA = False
            memory = Memory()

            while t < T:
                # print(t)

                state = (env, rec, context, context2, action)

                if self.model_params["use_LoRA"] and t >= self.trainer_params["LoRA_begin_step"]: use_LoRA = True
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    action, log_lh, improvement_method = self.model(state, solver="improvement", require_entropy=False, use_LoRA=use_LoRA)

                if self.model.training: memory.logprobs.append(log_lh.clone())
                # state transient
                # rec, rewards, obj, feasibility_history, context, context2, info
                rec, rewards, obj, feasibility_history, context, context2, info, out_penalty, out_node_penalty \
                    = env.improvement_step(rec, action, obj, feasibility_history, t,
                                           improvement_method = improvement_method, insert_before=self.trainer_params["insert_before"],
                                           weights=weights, out_reward=self.trainer_params["out_reward"],
                                           n2s_decoder=self.model_params["n2s_decoder"])

                # update best solution
                criterion = obj[:, 0].clone().view(batch_size // aug_factor, aug_factor * pomo_size)
                tmp_fsb = (out_penalty.sum(0) <= 0.0).clone().view(batch_size // aug_factor, aug_factor * pomo_size)
                new_best, best_index = criterion.masked_fill(~tmp_fsb, float("inf")).min(-1)
                no_feasible = torch.isinf(new_best)
                if no_feasible.any():
                    fallback_best, fallback_index = criterion[no_feasible, :].min(dim=-1)
                    new_best[no_feasible] = fallback_best
                    best_index[no_feasible] = fallback_index
                index = new_best < best_reward
                is_improved = (is_improved | index)
                rec_best[index] = rec.view(batch_size // aug_factor, aug_factor * pomo_size, -1)[torch.arange(batch_size // aug_factor),best_index, :][index].clone()  # update best solution

                if self.model.training: batch_reward.append(rewards[:, 0].clone())
                memory.rewards.append(rewards)
                criterion = obj.clone() if not self.trainer_params["seperate_obj_penalty"] else (obj[0] + obj[1]).clone()
                memory.obj.append(criterion.clone())
                rec_iter_history = torch.cat([rec_iter_history, rec.unsqueeze(1)], dim=1) # b, 1~t+1, solution_length

                memory.out_node_penalty.append(out_node_penalty)  # (c, b*k)
                memory.out_penalty.append(out_penalty)  # (c, b*k)
                feasible = out_penalty.sum(0) <= 0.0
                soft_infeasible = (out_penalty.sum(0) <= epsilon) & (out_penalty.sum(0) > 0.)
                memory.feasible.append(feasible)
                memory.soft_feasible.append(soft_infeasible)

                # feasible = out_penalty.sum(0) <= 0.0
                # feasible_all += feasible.int()
                # # Update scores with current step's results
                # min_scores = torch.where(feasible, torch.min(min_scores, obj[:, 0]), min_scores).clone()
                # out_node_penalties = torch.min(out_node_penalties, out_node_penalty.sum(0)).clone()
                # out_penalties = torch.min(out_penalties, out_penalty.sum(0)).clone()

                # next
                t = t + 1

            # calculate improvement loss
            if self.model.training:
                log_prob = torch.stack(memory.logprobs).view(T, batch_size, pomo_size)
                reward = torch.stack(memory.rewards).sum(-1).view(T, batch_size, pomo_size)
                baseline = 0. if pomo_size == 1 else reward.mean(-1, keepdims=True)
                advantage = reward - baseline
                loss = - advantage * log_prob  # Minus Sign: To Increase REWARD
                loss_mean = loss.mean(-1).mean(0) # shape: (aug_batch,)

            # update model
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            # prepare solution for the next iter
            feasible_all = torch.cat([feasible_all.unsqueeze(0), torch.stack(memory.feasible)], dim=0) # shape: (T+1, batch*pomo)
            score = torch.cat((score.unsqueeze(0), torch.stack(memory.obj)), dim=0) # shape: (T+1, batch*pomo)
            criterion_ = score[:, :, 0].clone().masked_fill(~feasible_all, float("inf"))# shape: (T+1, batch*pomo)
            new_best_, best_index_ = criterion_.min(0)
            no_feasible = torch.isinf(new_best_)
            if no_feasible.any():
                fallback_best, fallback_index = score[:, :, 0][:, no_feasible].min(dim=0)
                new_best_[no_feasible] = fallback_best
                best_index_[no_feasible] = fallback_index
            index_ = new_best_ < best_reward_iter
            best_reward_iter[index_] = new_best_[index_]  # update best reward
            rec_best_iter[index_] = rec_iter_history[torch.arange(batch_size), best_index_, :][index_].clone()  # update best solution
            solution = remove_dummy_depot_from_solution(rec2sol(rec_best_iter).view(batch_size, 1, -1), env.problem_size) # no need to rm redundant depots
            feasible_best = feasible_all.any(0) # shape: (batch*pomo)

            memory.clear_memory()


        # calculate infeasible outputs (BSF during construction+improvement)
        feasible = feasible_best # shape: (aug*batch*pomo)
        aug_feasible = feasible.reshape(aug_factor, -1, pomo_size).any(dim=0).any(dim=-1) # shape: (batch,)
        no_aug_feasible = feasible.reshape(aug_factor, -1, pomo_size)[0].any(dim=-1) # shape: (batch,)
        self.val_metric_logger._improve_tensor_update("aug_feasible", aug_feasible)
        self.val_metric_logger._improve_tensor_update("no_aug_feasible", no_aug_feasible)
        sol_infeasible_rate = 1. - (feasible.sum() / (batch_size * pomo_size)) # batch_size = aug*batch
        ins_infeasible_rate = 1. - aug_feasible.sum() / (batch_size//aug_factor)
        self.val_metric_logger.improve_metrics["sol_infeasible_rate"].update(sol_infeasible_rate, batch_size * pomo_size)
        self.val_metric_logger.improve_metrics["ins_infeasible_rate"].update(ins_infeasible_rate, batch_size // aug_factor)

        # score = cost
        min_scores = best_reward_iter.view(aug_factor, -1, pomo_size)
        aug_score_fsb = min_scores.min(dim=-1)[0].min(dim=0)[0]
        no_aug_score_fsb = min_scores[0].min(dim=-1)[0]
        self.val_metric_logger._improve_tensor_update("aug_score", aug_score_fsb)
        self.val_metric_logger._improve_tensor_update("no_aug_score", no_aug_score_fsb)

        # penalty
        # out_node_penalties = out_node_penalties.view(aug_factor, -1, pomo_size)
        # no_aug_out_nodes = out_node_penalties[0].min(dim=-1)[0]
        # aug_out_nodes = out_node_penalties.min(dim=0)[0].min(dim=-1)[0]
        # out_penalties = out_penalties.view(aug_factor, -1, pomo_size)
        # no_aug_out = out_penalties[0].min(dim=-1)[0]
        # aug_out = out_penalties.min(dim=0)[0].min(dim=-1)[0]
        # self.val_metric_logger._improve_tensor_update("no_aug_out", no_aug_out)
        # self.val_metric_logger._improve_tensor_update("no_aug_out_nodes", no_aug_out_nodes)
        # self.val_metric_logger._improve_tensor_update("aug_out", aug_out)
        # self.val_metric_logger._improve_tensor_update("aug_out_nodes", aug_out_nodes)

        # end update
        # memory.clear_memory()

        return (aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible,
                remove_dummy_depot_from_solution(rec2sol(rec_best).view(batch_size//aug_factor, 1, -1), env.problem_size), is_improved)

    def _val_reimprovement(self, env, aug_factor, old_aug_score_fsb, old_no_aug_score_fsb, old_aug_feasible, old_no_aug_feasible):
        with torch.no_grad():
            # solution/rec shape: (batch, pomo, solution)
            solution = env.selected_node_list.clone()
            if "TSP" not in self.args.problem: solution = get_solution_with_dummy_depot(solution, env.problem_size)
            batch_size, pomo_size, solution_size = solution.size()  # batch_size = aug_factor * batch_size
            rec = sol2rec(solution).view(batch_size * pomo_size, -1)
            # preapare input
            obj, context, out_penalty, out_node_penalty = env.get_costs(rec, get_context=True)  # obj only
            obj = torch.cat((obj[:, None], obj[:, None], obj[:, None]), -1).clone()
            # obj = obj.unsqueeze(-1).expand(-1, -1, 3)
            total_history = self.trainer_params["total_history"]
            if self.model_params["n2s_decoder"]:
                context2 = torch.zeros(batch_size * pomo_size, 4, solution_size)
                feasibility_history = torch.zeros(batch_size * pomo_size, total_history, solution_size)
            else:
                context2 = torch.zeros(batch_size * pomo_size, 9)
                context2[:, -1] = (~env.infeasible).view(-1)  # current feasibility
                feasibility_history = (~env.infeasible).view(-1, 1).expand(batch_size * pomo_size, total_history)
            action = None
            best_reward, best_index = (obj[:, 0].view(batch_size // aug_factor, aug_factor * pomo_size)).min(-1)
            rec_best = rec.view(batch_size // aug_factor, aug_factor * pomo_size, -1)[
                       torch.arange(batch_size // aug_factor), best_index, :].clone()
            # print(f"!!!!!!!constructed best: {remove_dummy_depot_from_solution(rec2sol(rec_best).view(batch_size, 1, -1), env.problem_size)[:3]}")
            is_improved = torch.zeros(batch_size // aug_factor).bool()

            # sample trajectory
            t = 0
            T = self.trainer_params["validation_improve_steps"]
            use_LoRA = False
            # memory = Memory()
            # initial solution from construction
            feasible_all = ((~env.infeasible).view(-1)).int()
            min_scores = torch.full((batch_size * pomo_size,), float('inf'))
            min_scores = torch.where(feasible_all.bool(), obj[:, 0], min_scores).clone()
            out_node_penalties = torch.full((batch_size * pomo_size,), float('inf'))
            # if self.args.problem == "CVRP":
            #     out_node_penalty = (context[2] > 1.00001).sum(-1)
            # elif self.args.problem == "TSPTW":
            #     out_node_penalty = (torch.clamp_min(context[1] - context[-1], 0.0) > 1e-5).sum(-1) # (b*k)
            out_node_penalties = torch.where(~(feasible_all.bool()), out_node_penalty, out_node_penalties).clone()
            del out_node_penalty
            out_penalties = torch.full((batch_size * pomo_size,), float('inf'))
            # if self.args.problem == "CVRP":
            #     out_penalty = ((context[3] - 1.00001) * (context[3] > 1.00001)).sum(dim=1)
            # elif self.args.problem == "TSPTW":
            #     out_penalty = torch.clamp_min(context[1] - context[-1], 0.0).sum(-1)
            #     if self.trainer_params["penalty_normalize"]:
            #         try:
            #             out_penalty = out_penalty / context[-1][:, 0]
            #         except:
            #             out_penalty = out_penalty / context[-1][:, :1]
            out_penalties = torch.where(~(feasible_all.bool()), out_penalty, out_penalties).clone()
            del out_penalty

            while t < T:
                # print(t)

                state = (env, rec, context, context2, action)

                if self.model_params["use_LoRA"] and t >= self.trainer_params["LoRA_begin_step"]: use_LoRA = True
                action, _, improvement_method = self.model(state, solver="improvement", require_entropy=False, use_LoRA=use_LoRA)

                # state transient
                # rec, rewards, obj, feasibility_history, context, context2, info
                rec, _, obj, feasibility_history, context, context2, _, out_penalty, out_node_penalty = env.improvement_step(
                    rec, action, obj, feasibility_history, t,
                    improvement_method=improvement_method, insert_before=self.trainer_params["insert_before"],
                    n2s_decoder=self.model_params["n2s_decoder"])
                # update best solution
                criterion = obj.clone()
                new_best, best_index = criterion[:, 0].view(batch_size // aug_factor, aug_factor * pomo_size).min(-1)
                index = new_best < best_reward
                best_reward[index] = new_best[index]  # update best reward
                is_improved = (is_improved | index)
                rec_best[index] = \
                    rec.view(batch_size // aug_factor, aug_factor * pomo_size, -1)[
                    torch.arange(batch_size // aug_factor),
                    best_index, :][index].clone()  # update best solution

                feasible = out_penalty.sum(0) <= 0.0
                feasible_all += feasible.int()
                # Update scores with current step's results
                min_scores = torch.where(feasible, torch.min(min_scores, obj[:, 0]), min_scores).clone()

                out_node_penalties = torch.min(out_node_penalties, out_node_penalty.sum(0)).clone()
                out_penalties = torch.min(out_penalties, out_penalty.sum(0)).clone()

                # next
                t = t + 1

            # calculate infeasible outputs (BSF during improvement)
            feasible = feasible_all > 0  # shape: (aug*batch*pomo)
            aug_feasible = feasible.reshape(aug_factor, -1, pomo_size).any(dim=0).any(dim=-1)  # shape: (batch,)
            no_aug_feasible = feasible.reshape(aug_factor, -1, pomo_size)[0].any(dim=-1)  # shape: (batch,)
            no_aug_feasible = no_aug_feasible | old_no_aug_feasible
            aug_feasible = aug_feasible | old_aug_feasible
            self.val_metric_logger._reimprove_tensor_update("aug_feasible", aug_feasible)
            self.val_metric_logger._reimprove_tensor_update("no_aug_feasible", no_aug_feasible)
            sol_infeasible_rate = 1. - (feasible.sum() / (batch_size * pomo_size))  # batch_size = aug*batch
            ins_infeasible_rate = 1. - aug_feasible.sum() / (batch_size // aug_factor)
            self.val_metric_logger.reimprove_metrics["sol_infeasible_rate"].update(sol_infeasible_rate,
                                                                                   batch_size * pomo_size)
            self.val_metric_logger.reimprove_metrics["ins_infeasible_rate"].update(ins_infeasible_rate,
                                                                                   batch_size // aug_factor)

            min_scores = min_scores.view(aug_factor, -1, pomo_size)
            aug_score_fsb = min_scores.min(dim=-1)[0].min(dim=0)[0]
            no_aug_score_fsb = min_scores[0].min(dim=-1)[0]
            no_aug_score_fsb = torch.min(no_aug_score_fsb, old_no_aug_score_fsb)
            aug_score_fsb = torch.min(aug_score_fsb, old_aug_score_fsb)
            self.val_metric_logger._reimprove_tensor_update("aug_score", aug_score_fsb)
            self.val_metric_logger._reimprove_tensor_update("no_aug_score", no_aug_score_fsb)

            return aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible, remove_dummy_depot_from_solution(
                rec2sol(rec_best).view(batch_size // aug_factor, 1, -1), env.problem_size), is_improved

    def _get_construction_output(self, infeasible, reward, prob_list=None, improve_reward=None, select_idx=None, probs_return_list=None, epsilon=None, bl_construct_detach=None, bl_construct=None):
        # Input.shape
        # infeasible: (batch, pomo)
        # reward (list or tensor): (batch, pomo)
        # prob_list (list or tensor): (batch, pomo, solution)
        # improve_reward (tensor): (batch)
        batch_size, pomo_size = infeasible.size()

        infeasible_output = infeasible
        if isinstance(reward, list):
            try:
                dist_reward, total_timeout_reward, timeout_nodes_reward = reward
            except:
                dist_reward, total_timeout_reward, timeout_nodes_reward, total_out_of_dl_reward, out_of_dl_nodes_reward, total_out_of_capacity_reward, out_of_capacity_nodes_reward = reward
            dist = dist_reward.clone()
        else:
            dist_reward = reward
            dist = reward

        # Calculate Feasibility Output
        if self.trainer_params["fsb_dist_only"]:
            problem_size = self.env_params["problem_size"]
            if self.trainer_params["train_z_sample_size"] != 0 and self.model_params["polynet"]:
                pomo_size = self.trainer_params["train_z_sample_size"]
            if self.trainer_params["infsb_dist_penalty"]:
                dist = torch.where(infeasible, -problem_size, dist_reward)
                # turn the dist_reward of the infeasible solutions into -problem_size (negative reward)
            if infeasible is None:
                infeasible = (timeout_nodes_reward + out_of_dl_nodes_reward + out_of_capacity_nodes_reward != 0.)
            feasible_number = (batch_size*pomo_size) - infeasible.sum()
            feasible_dist_mean, feasible_dist_max_pomo_mean = 0., 0.

            batch_feasible = (infeasible == False).any(dim=-1)  # shape: (batch)
            self.metric_logger.construct_metrics["sol_infeasible_rate"].update(infeasible.sum() / (batch_size * pomo_size), batch_size)
            self.metric_logger.construct_metrics["ins_infeasible_rate"].update(1. - batch_feasible.sum() / batch_size, batch_size)

            if feasible_number:
                feasible_dist = torch.where(infeasible, torch.zeros_like(dist_reward), dist_reward) # feasible dist left only
                feasible_dist_mean = -feasible_dist.sum() / feasible_number # negative sign to make positive value, and calculate mean

                reward_masked = dist.masked_fill(infeasible, -1e10)  # get feasible results from pomo
                feasible_max_pomo_dist = reward_masked.max(dim=1)[0]# get best results from pomo, shape: (batch)
                # max negative obj. is equal to min obj
                # feasible_max_pomo_dist = dist.max(dim=1)[0] # get best results from pomo, shape: (batch)
                feasible_max_pomo_dist = torch.where(batch_feasible==False, torch.zeros_like(feasible_max_pomo_dist), feasible_max_pomo_dist) # feasible dist left only
                feasible_dist_max_pomo_mean = -feasible_max_pomo_dist.sum() / batch_feasible.sum() # negative sign to make positive value, and calculate mean
                self.metric_logger.construct_metrics["feasible_dist_mean"].update(feasible_dist_mean, feasible_number)
                self.metric_logger.construct_metrics["feasible_dist_max_pomo_mean"].update(feasible_dist_max_pomo_mean, batch_feasible.sum())

        # Calculate Loss
        if prob_list is not None:
            if self.model_params["dual_decoder"]:
                assert isinstance(prob_list, list), "Prob lists of each decoder should be imported!"
                prob_list1, prob_list2 = prob_list

                assert self.trainer_params["baseline"] == "group", "Only group baseline is supported!"

                weight1, weight2 = 1, 1
                dist_more_reward = weight1 * dist + weight2 * (total_timeout_reward + timeout_nodes_reward)
                # dist_more_reward = dist + total_timeout_reward + timeout_nodes_reward
                advantage1 = dist_more_reward - dist_more_reward.float().mean(dim=1, keepdims=True)  # (batch, pomo)
                log_prob1 = prob_list1.log().sum(dim=2)
                loss1 = -advantage1 * log_prob1  # Minus Sign: To Increase REWARD

                timeout_more_reward = weight2 * dist + weight1 *(total_timeout_reward + timeout_nodes_reward)
                # timeout_more_reward = dist + total_timeout_reward + timeout_nodes_reward
                advantage2 = timeout_more_reward - timeout_more_reward.float().mean(dim=1, keepdims=True)  # (batch, pomo)
                log_prob2 = prob_list2.log().sum(dim=2)
                loss2 = -advantage2 * log_prob2  # Minus Sign: To Increase REWARD
                self.metric_logger.construct_metrics["loss1"].update(loss1.mean().item(), batch_size)
                self.metric_logger.construct_metrics["loss2"].update(loss2.mean().item(), batch_size)

                kl_loss = (prob_list1 * (prob_list1.log() - prob_list2.log())).sum(-1)
                loss_mean = loss1.mean()/loss1.mean().detach() + loss2.mean()/loss2.mean().detach() - kl_loss.mean()/kl_loss.mean().detach()
            else:
                if isinstance(reward, list):
                    try:
                        if self.trainer_params["subgradient"]:
                            reward = dist + self.lambda_[0] * (total_timeout_reward + timeout_nodes_reward) + self.lambda_[3] * (total_out_of_dl_reward + out_of_dl_nodes_reward) + self.lambda_[1] * (total_out_of_capacity_reward + out_of_capacity_nodes_reward)
                        else:
                            if self.trainer_params["non_linear_cons"]:
                                penalty = (total_timeout_reward + timeout_nodes_reward +
                                           total_out_of_dl_reward + out_of_dl_nodes_reward +
                                           total_out_of_capacity_reward + out_of_capacity_nodes_reward)
                                if self.trainer_params["non_linear"] == "scalarization":
                                    reward = ((penalty) / (dist + penalty)) * dist + ((dist) / (dist + penalty)) * penalty
                                elif self.trainer_params["non_linear"] in ["fixed_epsilon", "decayed_epsilon"]:
                                    reward = dist + self.penalty_factor * penalty
                                    reward = torch.where(-penalty > epsilon, 10 * reward, reward) # todo: 10 is hardcoded [NOW CORRECT]
                                else:
                                    raise NotImplementedError
                            else:
                                reward = dist + self.penalty_factor * (total_timeout_reward + timeout_nodes_reward +
                                                                       total_out_of_dl_reward + out_of_dl_nodes_reward +
                                                                       total_out_of_capacity_reward + out_of_capacity_nodes_reward)
                    except:
                        if self.trainer_params["wo_node_penalty"]:
                            reward = dist + self.penalty_factor * (total_timeout_reward)
                        elif self.trainer_params["wo_tour_penalty"]:
                            reward = dist + self.penalty_factor * (timeout_nodes_reward)
                        else:
                            reward = dist +  self.penalty_factor * (total_timeout_reward +  timeout_nodes_reward)  # (batch, pomo)
                if not self.trainer_params["out_reward"] and self.trainer_params["fsb_reward_only"]: #ATTENTION
                    if self.trainer_params["shared_critic"]:
                        raise NotImplementedError
                    elif self.trainer_params["baseline"] == "group":
                        feasible_reward_number = (infeasible==False).sum(-1)
                        feasible_reward_mean = (torch.where(infeasible, torch.zeros_like(dist_reward), dist_reward).sum(-1) / feasible_reward_number)[:,None]
                        baseline = feasible_reward_mean
                    elif self.trainer_params["baseline"] == "improve":
                        # improve_reward.shape: (batch)
                        baseline = improve_reward.float().mean(dim=0)
                    else:
                        raise NotImplementedError
                    feasible_advantage = dist_reward - baseline # (batch, pomo)
                    feasible_advantage = torch.masked_select(feasible_advantage, infeasible==False)
                    log_prob = torch.masked_select(prob_list.log().sum(dim=2), infeasible==False)
                    advantage = feasible_advantage
                else:
                    if self.trainer_params["shared_critic"]:
                        reward = torch.gather(reward, dim=1, index=select_idx)
                        log_prob = prob_list.log().sum(dim=2)  # (batch, pomo)
                        log_prob = torch.gather(log_prob, dim=1, index=select_idx)
                        advantage = (reward - bl_construct_detach.view(reward.size(0), -1)).detach() / 1000.
                        # print(reward.mean(), bl_construct_detach.mean(), advantage.mean())
                    elif self.trainer_params["baseline"] == "group":
                        baseline = reward.float().mean(dim=1, keepdims=True)
                        advantage = reward - baseline  # (batch, pomo)
                        if self.trainer_params["bonus_for_construction"] and improve_reward is not None:
                            after_improve_reward = -reward.clone()
                            after_improve_reward.scatter_(dim=1, index=select_idx, src=improve_reward)
                            new_advantage = advantage / 10
                            new_advantage = torch.where(after_improve_reward < -reward, advantage, new_advantage)
                            advantage = new_advantage.clone()
                        if self.trainer_params["extra_bonus"] and improve_reward is not None:
                            improvement_value = (-reward[torch.arange(batch_size)[:, None], select_idx] - improve_reward).clamp(min=0) # batch_size * k
                            advantage[torch.arange(batch_size)[:, None], select_idx] += improvement_value * self.trainer_params["extra_weight"]
                            self.metric_logger.construct_metrics["improvement_value"].update(improvement_value.mean().item(), batch_size)
                        log_prob = prob_list.log().sum(dim=2)  # (batch, pomo)
                    elif self.trainer_params["baseline"] == "improve":
                        # improve_reward.shape: (batch)
                        baseline = -improve_reward.float()
                        advantage = reward - baseline  # (batch, pomo)
                        log_prob = prob_list.log().sum(dim=2)  # (batch, pomo)
                    elif self.trainer_params["baseline"] == "share":
                        full_index = torch.arange(pomo_size).expand(batch_size, pomo_size)
                        mask = torch.zeros_like(full_index, dtype=torch.bool)
                        mask[torch.arange(batch_size).unsqueeze(1), select_idx] = True
                        non_select_idx = full_index[~mask].view(batch_size, -1)
                        advantage = reward.gather(1, non_select_idx) - reward.gather(1, non_select_idx).float().mean(dim=1, keepdims=True)  # (batch, pomo-k)
                        log_prob = prob_list.log().sum(dim=2).gather(1, non_select_idx)  # (batch, pomo-k)
                    else:
                        raise NotImplementedError
                loss = - advantage * log_prob  # Minus Sign: To Increase REWARD
                if self.trainer_params["shared_critic"]:
                    self.metric_logger.construct_metrics["construct_RL_loss"].update(loss.mean().item(), batch_size)
                    baseline_loss =  (((bl_construct.view(reward.size(0), -1) - reward.detach()) / 1000.) ** 2)
                    self.metric_logger.construct_metrics["critic_loss"].update(baseline_loss.mean().item(),batch_size)
                    loss = loss + baseline_loss
                if self.trainer_params["diversity_loss"]:
                    self.metric_logger.construct_metrics["construct_RL_loss"].update(loss.mean().item(), batch_size)
                    if probs_return_list is None:
                        # implementation 1: only focus on the entropy on the probs of the selected nodes
                        diversity_loss = -(prob_list * prob_list.log()).sum(dim=2)  # Entropy
                    else:
                        # implementation 2: increase diversity for the whole action probability distributions
                        diversity_loss = -(probs_return_list * probs_return_list.log()).sum(dim=2).mean(dim=-1) # b * p
                    if self.trainer_params["shared_critic"]:
                        loss = loss.mean() - self.trainer_params['diversity_weight'] * diversity_loss.mean()
                    else:
                        loss = loss - self.trainer_params['diversity_weight'] * diversity_loss  # Minus Sign: To Increase Diversity (i.e. Entropy)
                    self.metric_logger.construct_metrics["diversity_loss"].update(diversity_loss.mean().item(), batch_size)
                loss_mean = loss.mean()
            self.metric_logger.construct_metrics["loss"].update(loss_mean.item(), batch_size)

        # aux_loss in MvMOE
        # if hasattr(self.model, "aux_loss"):
        #     loss_mean = loss_mean + self.model.aux_loss  # add aux(moe)_loss for load balancing (default coefficient: 1e-2)

        # Calculate scores (tour length)
        if not self.trainer_params["out_reward"]:
            max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
            score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
            self.metric_logger.construct_metrics["score"].update(score_mean.item(), batch_size)
        else:
            # note: different from improvement output
            max_dist_reward = dist_reward.max(dim=1)[0]  # get best results from pomo
            dist_mean = -max_dist_reward.float().mean()  # negative sign to make positive value
            max_timeout_reward = total_timeout_reward.max(dim=1)[0]  # get best results from pomo
            timeout_mean = -max_timeout_reward.float().mean()  # negative sign to make positive value
            max_timeout_nodes_reward = timeout_nodes_reward.max(dim=1)[0]  # get best results from pomo
            timeout_nodes_mean = -max_timeout_nodes_reward.float().mean()  # negative sign to make positive value
            self.metric_logger.construct_metrics["score"].update(dist_mean, batch_size)
            self.metric_logger.construct_metrics["out"].update(timeout_mean, batch_size)
            self.metric_logger.construct_metrics["out_nodes"].update(timeout_nodes_mean, batch_size)
            if self.args.problem in ["OVRPBLTW", "OVRPLTW", "VRPBLTW"]:
                max_dl_reward = total_out_of_dl_reward.max(dim=1)[0]  # get best results from pomo
                dlout_mean = -max_dl_reward.float().mean()  # negative sign to make positive value
                max_dl_nodes_reward = out_of_dl_nodes_reward.max(dim=1)[0]  # get best results from pomo
                dlout_nodes_mean = -max_dl_nodes_reward.float().mean()  # negative sign to make positive value
                max_capacity_reward = total_out_of_capacity_reward.max(dim=1)[0]  # get best results from pomo
                capacity_out_mean = -max_capacity_reward.float().mean()  # negative sign to make positive value
                max_capacity_nodes_reward = out_of_capacity_nodes_reward.max(dim=1)[0]  # get best results from pomo
                capacity_out_nodes_mean = -max_capacity_nodes_reward.float().mean()  # negative sign to make positive value
                self.metric_logger.construct_metrics["dlout"].update(dlout_mean, batch_size)
                self.metric_logger.construct_metrics["dlout_nodes"].update(dlout_nodes_mean, batch_size)
                self.metric_logger.construct_metrics["capacity_out"].update(capacity_out_mean, batch_size)
                self.metric_logger.construct_metrics["capacity_out_nodes"].update(capacity_out_nodes_mean, batch_size)

        if prob_list is not None: return loss_mean

    def _get_reconstruction_output(self, infeasible, reward, prob_list=None, probs_return_list=None, epsilon=None, improve_reward=None):
        # Input.shape
        # infeasible: (batch, pomo)
        # reward (list or tensor): (batch, pomo)
        # prob_list (list or tensor): (batch, pomo, solution)
        batch_size, pomo_size = infeasible.size()

        infeasible_output = infeasible
        if isinstance(reward, list):
            try:
                dist_reward, total_timeout_reward, timeout_nodes_reward = reward
            except:
                dist_reward, total_timeout_reward, timeout_nodes_reward, total_out_of_dl_reward, out_of_dl_nodes_reward, total_out_of_capacity_reward, out_of_capacity_nodes_reward = reward
            dist = dist_reward.clone()
        else:
            dist_reward = reward
            dist = reward

        # Calculate Feasibility Output
        if self.trainer_params["fsb_dist_only"]:
            problem_size = self.env_params["problem_size"]
            if self.trainer_params["train_z_sample_size"] != 0 and self.model_params["polynet"]:
                pomo_size = self.trainer_params["train_z_sample_size"]
            if self.trainer_params["infsb_dist_penalty"]:
                dist = torch.where(infeasible, -problem_size, dist_reward)
                # turn the dist_reward of the infeasible solutions into -problem_size (negative reward)
            if infeasible is None:
                infeasible = (timeout_nodes_reward + out_of_dl_nodes_reward + out_of_capacity_nodes_reward != 0.)
            feasible_number = (batch_size*pomo_size) - infeasible.sum()
            feasible_dist_mean, feasible_dist_max_pomo_mean = 0., 0.

            batch_feasible = (infeasible == False).any(dim=-1)  # shape: (batch)
            self.metric_logger.reconstruct_metrics["sol_infeasible_rate"].update(infeasible.sum() / (batch_size * pomo_size), batch_size)
            self.metric_logger.reconstruct_metrics["ins_infeasible_rate"].update(1. - batch_feasible.sum() / batch_size, batch_size)

            if feasible_number:
                feasible_dist = torch.where(infeasible, torch.zeros_like(dist_reward), dist_reward) # feasible dist left only
                feasible_dist_mean = -feasible_dist.sum() / feasible_number # negative sign to make positive value, and calculate mean

                reward_masked = dist.masked_fill(infeasible, -1e10)  # get feasible results from pomo
                feasible_max_pomo_dist = reward_masked.max(dim=1)[0]# get best results from pomo, shape: (batch)
                # max negative obj. is equal to min obj
                # feasible_max_pomo_dist = dist.max(dim=1)[0] # get best results from pomo, shape: (batch)
                feasible_max_pomo_dist = torch.where(batch_feasible==False, torch.zeros_like(feasible_max_pomo_dist), feasible_max_pomo_dist) # feasible dist left only
                feasible_dist_max_pomo_mean = -feasible_max_pomo_dist.sum() / batch_feasible.sum() # negative sign to make positive value, and calculate mean
                self.metric_logger.reconstruct_metrics["feasible_dist_mean"].update(feasible_dist_mean, feasible_number)
                self.metric_logger.reconstruct_metrics["feasible_dist_max_pomo_mean"].update(feasible_dist_max_pomo_mean, batch_feasible.sum())

        # Calculate Loss
        if prob_list is not None:
            if isinstance(reward, list):
                try:
                    if self.trainer_params["subgradient"]:
                        reward = dist + self.lambda_[0] * (total_timeout_reward + timeout_nodes_reward) + self.lambda_[3] * (total_out_of_dl_reward + out_of_dl_nodes_reward) + self.lambda_[1] * (total_out_of_capacity_reward + out_of_capacity_nodes_reward)
                    else:
                        if self.trainer_params["non_linear_cons"]:
                            penalty = (total_timeout_reward + timeout_nodes_reward +
                                       total_out_of_dl_reward + out_of_dl_nodes_reward +
                                       total_out_of_capacity_reward + out_of_capacity_nodes_reward)
                            if self.trainer_params["non_linear"] == "scalarization":
                                reward = ((penalty) / (dist + penalty)) * dist + ((dist) / (dist + penalty)) * penalty
                            elif self.trainer_params["non_linear"] in ["fixed_epsilon", "decayed_epsilon"]:
                                reward = dist + self.penalty_factor * penalty
                                reward = torch.where(-penalty > epsilon, 10 * reward, reward) # todo: 10 is hardcoded [NOW CORRECT]
                            else:
                                raise NotImplementedError
                        else:
                            reward = dist + self.penalty_factor * (total_timeout_reward + timeout_nodes_reward +
                                                                   total_out_of_dl_reward + out_of_dl_nodes_reward +
                                                                   total_out_of_capacity_reward + out_of_capacity_nodes_reward)
                except:
                    reward = dist +  self.penalty_factor * (total_timeout_reward +  timeout_nodes_reward)  # (batch, pomo)
            if not self.trainer_params["out_reward"] and self.trainer_params["fsb_reward_only"]: #ATTENTION
                if self.trainer_params["baseline"] == "group":
                    feasible_reward_number = (infeasible==False).sum(-1)
                    feasible_reward_mean = (torch.where(infeasible, torch.zeros_like(dist_reward), dist_reward).sum(-1) / feasible_reward_number)[:,None]
                    baseline = feasible_reward_mean
                else:
                    raise NotImplementedError
                feasible_advantage = dist_reward - baseline # (batch, pomo)
                feasible_advantage = torch.masked_select(feasible_advantage, infeasible==False)
                log_prob = torch.masked_select(prob_list.log().sum(dim=2), infeasible==False)
                advantage = feasible_advantage
            else:
                if self.trainer_params["baseline"] == "group":
                    baseline = reward.float().mean(dim=1, keepdims=True)
                    advantage = reward - baseline  # (batch, pomo)
                    if self.trainer_params["reconstruct_improve_bonus"]:
                        advantage += improve_reward - improve_reward.float().mean(dim=1, keepdims=True)
                    log_prob = prob_list.log().sum(dim=2)  # (batch, pomo)
                else:
                    raise NotImplementedError
            loss = - advantage * log_prob  # Minus Sign: To Increase REWARD
            if self.trainer_params["diversity_loss"]:
                self.metric_logger.reconstruct_metrics["construct_RL_loss"].update(loss.mean().item(), batch_size)
                if probs_return_list is None:
                    # implementation 1: only focus on the entropy on the probs of the selected nodes
                    diversity_loss = -(prob_list * prob_list.log()).sum(dim=2)  # Entropy
                else:
                    # implementation 2: increase diversity for the whole action probability distributions
                    diversity_loss = -(probs_return_list * probs_return_list.log()).sum(dim=2).mean(dim=-1) # b * p
                loss = loss - self.trainer_params['diversity_weight'] * diversity_loss  # Minus Sign: To Increase Diversity (i.e. Entropy)
                self.metric_logger.reconstruct_metrics["diversity_loss"].update(diversity_loss.mean().item(), batch_size)
            loss_mean = loss.mean()
            self.metric_logger.reconstruct_metrics["loss"].update(loss_mean.item(), batch_size)

        # Calculate scores (tour length)
        if not self.trainer_params["out_reward"]:
            max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
            score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
            self.metric_logger.reconstruct_metrics["score"].update(score_mean.item(), batch_size)
        else:
            max_dist_reward = dist_reward.max(dim=1)[0]  # get best results from pomo
            dist_mean = -max_dist_reward.float().mean()  # negative sign to make positive value
            max_timeout_reward = total_timeout_reward.max(dim=1)[0]  # get best results from pomo
            timeout_mean = -max_timeout_reward.float().mean()  # negative sign to make positive value
            max_timeout_nodes_reward = timeout_nodes_reward.max(dim=1)[0]  # get best results from pomo
            timeout_nodes_mean = -max_timeout_nodes_reward.float().mean()  # negative sign to make positive value
            self.metric_logger.reconstruct_metrics["score"].update(dist_mean, batch_size)
            self.metric_logger.reconstruct_metrics["out"].update(timeout_mean, batch_size)
            self.metric_logger.reconstruct_metrics["out_nodes"].update(timeout_nodes_mean, batch_size)
            if self.args.problem in ["OVRPBLTW", "OVRPLTW", "VRPBLTW"]:
                max_dl_reward = total_out_of_dl_reward.max(dim=1)[0]  # get best results from pomo
                dlout_mean = -max_dl_reward.float().mean()  # negative sign to make positive value
                max_dl_nodes_reward = out_of_dl_nodes_reward.max(dim=1)[0]  # get best results from pomo
                dlout_nodes_mean = -max_dl_nodes_reward.float().mean()  # negative sign to make positive value
                max_capacity_reward = total_out_of_capacity_reward.max(dim=1)[0]  # get best results from pomo
                capacity_out_mean = -max_capacity_reward.float().mean()  # negative sign to make positive value
                max_capacity_nodes_reward = out_of_capacity_nodes_reward.max(dim=1)[0]  # get best results from pomo
                capacity_out_nodes_mean = -max_capacity_nodes_reward.float().mean()  # negative sign to make positive value
                self.metric_logger.reconstruct_metrics["dlout"].update(dlout_mean, batch_size)
                self.metric_logger.reconstruct_metrics["dlout_nodes"].update(dlout_nodes_mean, batch_size)
                self.metric_logger.reconstruct_metrics["capacity_out"].update(capacity_out_mean, batch_size)
                self.metric_logger.reconstruct_metrics["capacity_out_nodes"].update(capacity_out_nodes_mean, batch_size)

        if prob_list is not None: return loss_mean

    def _get_construction_output_val(self, aug_factor, infeasible, reward, solution=None, incumbent_reward=None):
        if isinstance(reward, list):
            try:
                dist_reward, total_timeout_reward, timeout_nodes_reward = reward
                batch_size, pomo_size = total_timeout_reward.size()
                batch_size = batch_size // aug_factor
            except:
                dist_reward, total_timeout_reward, timeout_nodes_reward, total_out_of_dl_reward, out_of_dl_nodes_reward, total_out_of_capacity_reward, out_of_capacity_nodes_reward = reward
                batch_size, pomo_size = total_timeout_reward.size()
                batch_size = batch_size // aug_factor

                aug_total_out_of_dl_reward = total_out_of_dl_reward.reshape(aug_factor, batch_size, pomo_size)
                max_pomo_total_out_of_dl_reward, _ = aug_total_out_of_dl_reward.max(dim=2)
                no_aug_total_out_of_dl_score = -max_pomo_total_out_of_dl_reward[0, :].float()
                max_aug_pomo_total_out_of_dl_reward, _ = max_pomo_total_out_of_dl_reward.max(dim=0)
                aug_total_out_of_dl_score = -max_aug_pomo_total_out_of_dl_reward.float()

                aug_out_of_dl_nodes_reward = out_of_dl_nodes_reward.reshape(aug_factor, batch_size, pomo_size)
                max_pomo_out_of_dl_nodes_reward, _ = aug_out_of_dl_nodes_reward.max(dim=2)
                no_aug_out_of_dl_nodes_score = -max_pomo_out_of_dl_nodes_reward[0,:].float()
                max_aug_pomo_out_of_dl_nodes_reward, _ = max_pomo_out_of_dl_nodes_reward.max(dim=0)
                aug_out_of_dl_nodes_score = -max_aug_pomo_out_of_dl_nodes_reward.float()

                self.val_metric_logger._construct_tensor_update("no_aug_total_out_of_dl", no_aug_total_out_of_dl_score)
                self.val_metric_logger._construct_tensor_update("no_aug_out_of_dl_nodes", no_aug_out_of_dl_nodes_score)
                self.val_metric_logger._construct_tensor_update("aug_total_out_of_dl", aug_total_out_of_dl_score)
                self.val_metric_logger._construct_tensor_update("aug_out_of_dl_nodes", aug_out_of_dl_nodes_score)

                aug_total_out_of_capacity_reward = total_out_of_capacity_reward.reshape(aug_factor, batch_size, pomo_size)
                max_pomo_total_out_of_capacity_reward, _ = aug_total_out_of_capacity_reward.max(dim=2)
                no_aug_total_out_of_capacity_score = -max_pomo_total_out_of_capacity_reward[0, :].float()
                max_aug_pomo_total_out_of_capacity_reward, _ = max_pomo_total_out_of_capacity_reward.max(dim=0)
                aug_total_out_of_capacity_score = -max_aug_pomo_total_out_of_capacity_reward.float()

                aug_out_of_capacity_nodes_reward = out_of_capacity_nodes_reward.reshape(aug_factor, batch_size, pomo_size)
                max_pomo_out_of_capacity_nodes_reward, _ = aug_out_of_capacity_nodes_reward.max(dim=2)
                no_aug_out_of_capacity_nodes_score = -max_pomo_out_of_capacity_nodes_reward[0, :].float()
                max_aug_pomo_out_of_capacity_nodes_reward, _ = max_pomo_out_of_capacity_nodes_reward.max( dim=0)
                aug_out_of_capacity_nodes_score = -max_aug_pomo_out_of_capacity_nodes_reward.float()

                self.val_metric_logger._construct_tensor_update("no_aug_total_out_of_capacity", no_aug_total_out_of_capacity_score)
                self.val_metric_logger._construct_tensor_update("no_aug_out_of_capacity_nodes", no_aug_out_of_capacity_nodes_score)
                self.val_metric_logger._construct_tensor_update("aug_total_out_of_capacity", aug_total_out_of_capacity_score)
                self.val_metric_logger._construct_tensor_update("aug_out_of_capacity_nodes", aug_out_of_capacity_nodes_score)

            dist = dist_reward.clone()

            aug_total_timeout_reward = total_timeout_reward.reshape(aug_factor, batch_size, pomo_size)
            max_pomo_total_timeout_reward, _ = aug_total_timeout_reward.max(dim=2)
            no_aug_total_timeout_score = -max_pomo_total_timeout_reward[0, :].float()
            max_aug_pomo_total_timeout_reward, _ = max_pomo_total_timeout_reward.max(dim=0)
            aug_total_timeout_score = -max_aug_pomo_total_timeout_reward.float()

            aug_timeout_nodes_reward = timeout_nodes_reward.reshape(aug_factor, batch_size, pomo_size)
            max_pomo_timeout_nodes_reward, _ = aug_timeout_nodes_reward.max(dim=2)
            no_aug_timeout_nodes_score = -max_pomo_timeout_nodes_reward[0, :].float()
            max_aug_pomo_timeout_nodes_reward, _ = max_pomo_timeout_nodes_reward.max(dim=0)
            aug_timeout_nodes_score = -max_aug_pomo_timeout_nodes_reward.float()

            self.val_metric_logger._construct_tensor_update("no_aug_out", no_aug_total_timeout_score)
            self.val_metric_logger._construct_tensor_update("no_aug_out_nodes", no_aug_timeout_nodes_score)
            self.val_metric_logger._construct_tensor_update("aug_out", aug_total_timeout_score)
            self.val_metric_logger._construct_tensor_update("aug_out_nodes", aug_timeout_nodes_score)
        else:
            dist = reward

        batch_size, pomo_size = dist.size()
        batch_size = batch_size // aug_factor
        aug_reward = dist.reshape(aug_factor, batch_size, pomo_size)
        # shape: (augmentation, batch, pomo)
        if incumbent_reward is not None:
            aug_reward = incumbent_reward.reshape(aug_factor, batch_size) # shape: (augmentation, batch)
            no_aug_score = -aug_reward[0, :].float()  # negative sign to make positive value
            max_aug_pomo_reward, _ = aug_reward.max(dim=0)  # get best results from augmentation # shape: (batch,)
            aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value
            best_solution = None
            no_aug_feasible = aug_feasible = torch.ones((batch_size,)).bool()
            self.val_metric_logger._construct_tensor_update("no_aug_feasible", no_aug_feasible)
            self.val_metric_logger._construct_tensor_update("aug_feasible", aug_feasible)
            self.val_metric_logger.construct_metrics["sol_infeasible_rate"].update(torch.tensor([0]), batch_size)
            self.val_metric_logger.construct_metrics["ins_infeasible_rate"].update(torch.tensor([0]), batch_size)
        elif self.trainer_params["fsb_dist_only"]:
            # shape: (augmentation, batch, pomo)
            if infeasible is None:
                infeasible = (timeout_nodes_reward + out_of_dl_nodes_reward + out_of_capacity_nodes_reward != 0.)
            infeasible = infeasible.reshape(aug_factor, batch_size, pomo_size)
            no_aug_feasible = (infeasible[0, :, :] == False).any(dim=-1)
            aug_feasible = (infeasible == False).any(dim=0).any(dim=-1)
            self.val_metric_logger._construct_tensor_update("no_aug_feasible", no_aug_feasible)
            self.val_metric_logger._construct_tensor_update("aug_feasible", aug_feasible)

            sol_infsb_rate = infeasible.sum() / (batch_size * pomo_size * aug_factor)
            ins_infsb_rate = 1. - aug_feasible.sum() / batch_size
            self.val_metric_logger.construct_metrics["sol_infeasible_rate"].update(sol_infsb_rate, batch_size)
            self.val_metric_logger.construct_metrics["ins_infeasible_rate"].update(ins_infsb_rate, batch_size)

            reward_masked = aug_reward.masked_fill(infeasible, -1e10)
            fsb_no_aug = reward_masked[0,:,:].max(dim=1).values
            fsb_aug = reward_masked.max(dim=0).values.max(dim=-1).values
            no_aug_score, aug_score = -fsb_no_aug, -fsb_aug
            best_solution = None
            if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]:
                best_reward, best_index = (dist.reshape(batch_size, aug_factor*pomo_size)).min(-1)
                best_solution = solution.reshape(batch_size, aug_factor*pomo_size, -1)[torch.arange(batch_size), best_index, :].clone()
            elif self.tester_params["best_solution_path"] is not None:
                feasible_all = ~(infeasible.reshape(aug_factor * pomo_size, batch_size))
                dist_ = - dist.reshape(aug_factor * pomo_size, batch_size).clone()
                try:
                    penalty = total_timeout_reward + timeout_nodes_reward + total_out_of_dl_reward + out_of_dl_nodes_reward + total_out_of_capacity_reward + out_of_capacity_nodes_reward
                except:
                    penalty = total_timeout_reward + timeout_nodes_reward
                penalty = - penalty.reshape(aug_factor * pomo_size, batch_size)
                solution_ = solution.reshape(aug_factor * pomo_size, batch_size, -1)
                best_reward, best_feasible, best_solution = cal_best_aug_batch(dist_, penalty, feasible_all, solution_)
                self.val_metric_logger.best_reward = best_reward
                self.val_metric_logger.best_feasible = best_feasible
                self.val_metric_logger.best_solution = best_solution
                self.val_metric_logger.refinement_feasible_bsf_history = best_feasible.unsqueeze(-1)
                self.val_metric_logger.refinement_reward_bsf_history = best_reward.unsqueeze(-1)
                # self.val_metric_logger.refinement_solution_history = solution_[0].unsqueeze(1) # batch, 0~T, solution
        else:
            max_pomo_reward, _ = aug_reward.max(dim=2)
            no_aug_score = -max_pomo_reward[0, :].float()
            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
            aug_score = -max_aug_pomo_reward.float()
            infeasible_output = infeasible

        self.val_metric_logger._construct_tensor_update("no_aug_score", no_aug_score)
        self.val_metric_logger._construct_tensor_update("aug_score", aug_score)

        return aug_score, no_aug_score, aug_feasible, no_aug_feasible, best_solution

    def _get_reconstruction_output_val(self, aug_factor, infeasible, reward, old_aug_score_fsb, old_no_aug_score_fsb, old_aug_feasible, old_no_aug_feasible):
        if isinstance(reward, list):
            try:
                dist_reward, total_timeout_reward, timeout_nodes_reward = reward
                batch_size, pomo_size = total_timeout_reward.size()
                batch_size = batch_size // aug_factor
            except:
                dist_reward, total_timeout_reward, timeout_nodes_reward, total_out_of_dl_reward, out_of_dl_nodes_reward, total_out_of_capacity_reward, out_of_capacity_nodes_reward = reward
                batch_size, pomo_size = total_timeout_reward.size()
                batch_size = batch_size // aug_factor
            dist = dist_reward.clone()
        else:
            dist = reward
        batch_size, pomo_size = dist.size()
        batch_size = batch_size // aug_factor
        aug_reward = dist.reshape(aug_factor, batch_size, pomo_size)
        # shape: (augmentation, batch, pomo)
        if self.trainer_params["fsb_dist_only"]:
            # shape: (augmentation, batch, pomo)
            if infeasible is None:
                infeasible = (timeout_nodes_reward + out_of_dl_nodes_reward + out_of_capacity_nodes_reward != 0.)
            infeasible = infeasible.reshape(aug_factor, batch_size, pomo_size)
            no_aug_feasible = (infeasible[0, :, :] == False).any(dim=-1)
            aug_feasible = (infeasible == False).any(dim=0).any(dim=-1)
            no_aug_feasible = no_aug_feasible | old_no_aug_feasible
            aug_feasible = aug_feasible | old_aug_feasible
            self.val_metric_logger._reconstruct_tensor_update("no_aug_feasible", no_aug_feasible)
            self.val_metric_logger._reconstruct_tensor_update("aug_feasible", aug_feasible)

            sol_infsb_rate = infeasible.sum() / (batch_size * pomo_size * aug_factor) # note: this is the original sol_infeasible_rate!
            ins_infsb_rate = 1. - aug_feasible.sum() / batch_size # note: this is not the original ins_infeasible_rate!
            self.val_metric_logger.reconstruct_metrics["sol_infeasible_rate"].update(sol_infsb_rate, batch_size)
            self.val_metric_logger.reconstruct_metrics["ins_infeasible_rate"].update(ins_infsb_rate, batch_size)

            reward_masked = aug_reward.masked_fill(infeasible, -1e10)
            fsb_no_aug = reward_masked[0,:,:].max(dim=1).values
            fsb_aug = reward_masked.max(dim=0).values.max(dim=-1).values
            no_aug_score, aug_score = -fsb_no_aug, -fsb_aug
            no_aug_score = torch.min(no_aug_score, old_no_aug_score_fsb)
            aug_score = torch.min(aug_score, old_aug_score_fsb)
        else:
            raise NotImplementedError("not checked yet!")
            # max_pomo_reward, _ = aug_reward.max(dim=2)
            # no_aug_score = -max_pomo_reward[0, :].float()
            # max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
            # aug_score = -max_aug_pomo_reward.float()
            # infeasible_output = infeasible

        self.val_metric_logger._reconstruct_tensor_update("no_aug_score", no_aug_score)
        self.val_metric_logger._reconstruct_tensor_update("aug_score", aug_score)

        return aug_score, no_aug_score, aug_feasible, no_aug_feasible

    def _supplement_construction(self, aug_score_fsb, no_aug_score_fsb, aug_feasible, no_aug_feasible, aug_factor, infeasible, reward):
        # all feasible solutions
        if isinstance(reward, list):
            try:
                dist_reward, _, _ = reward
                batch_size, pomo_size = dist_reward.size()
                batch_size = batch_size // aug_factor
            except:
                dist_reward, _, _, _, _, _, _ = reward
                batch_size, pomo_size = dist_reward.size()
                batch_size = batch_size // aug_factor
            dist = dist_reward.clone()
        else:
            dist = reward

        batch_size, pomo_size = dist.size()
        batch_size = batch_size // aug_factor
        aug_reward = dist.reshape(aug_factor, batch_size, pomo_size)
        infeasible = infeasible.reshape(aug_factor, batch_size, pomo_size)
        # shape: (augmentation, batch, pomo)

        if self.trainer_params["fsb_dist_only"]:
            reward_masked = aug_reward.masked_fill(infeasible, -1e10)
            fsb_no_aug = reward_masked[0,:,:].max(dim=1).values
            fsb_aug = reward_masked.max(dim=0).values.max(dim=-1).values
            rc_no_aug_score, rc_aug_score = -fsb_no_aug, -fsb_aug
            # shape: (augmentation, batch, pomo)
            rc_no_aug_feasible = (infeasible[0, :, :] == False).any(dim=-1)
            rc_aug_feasible = (infeasible == False).any(dim=0).any(dim=-1)
            # merge the reconstrcuted feasible solutions with the original (construct-then-improve) ones
            no_aug_score_fsb = torch.where(rc_no_aug_feasible, torch.min(rc_no_aug_score, no_aug_score_fsb), no_aug_score_fsb)
            aug_score_fsb = torch.where(rc_aug_feasible, torch.min(rc_aug_score, aug_score_fsb), aug_score_fsb)
            aug_feasible = aug_feasible | rc_aug_feasible
            no_aug_feasible = no_aug_feasible | rc_no_aug_feasible
            self.val_metric_logger._reconstruct_masked_tensor_update("no_aug_feasible", no_aug_feasible)
            self.val_metric_logger._reconstruct_masked_tensor_update("aug_feasible", aug_feasible)
            sol_infsb_rate = infeasible.sum() / (batch_size * pomo_size * aug_factor)
            ins_infsb_rate = 1. - aug_feasible.sum() / batch_size
            self.val_metric_logger.reconstruct_masked_metrics["sol_infeasible_rate"].update(sol_infsb_rate, batch_size)
            self.val_metric_logger.reconstruct_masked_metrics["ins_infeasible_rate"].update(ins_infsb_rate, batch_size)
        else:
            raise NotImplementedError("Not checked yet")
            # max_pomo_reward, _ = aug_reward.max(dim=2)
            # no_aug_score = -max_pomo_reward[0, :].float()
            # max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
            # aug_score = -max_aug_pomo_reward.float()
            # infeasible_output = infeasible

        self.val_metric_logger._reconstruct_masked_tensor_update("no_aug_score", no_aug_score_fsb)
        self.val_metric_logger._reconstruct_masked_tensor_update("aug_score", aug_score_fsb)

    def _print_log(self):
        if self.model_params["pip_decoder"] and self.is_train_pip_decoder and self.metric_logger.construct_metrics["sl_loss"].count >0:
            if self.rank==0: print('Epoch {:3d}: PIP-D Loss: {:.4f},  Accuracy: {:.4f}% (BSF: {:.4f}%) [Infeasible: {:.4f}% (BSF: {:.4f}%), Feasible: {:.4f}% (BSF: {:.4f}%)]'.format(self.epoch, self.metric_logger.construct_metrics["sl_loss"].avg, self.metric_logger.construct_metrics["accuracy"].avg*100, self.accuracy_bsf*100, self.metric_logger.construct_metrics["infsb_accuracy"].avg*100, self.infsb_accuracy_bsf*100, self.metric_logger.construct_metrics["fsb_accuracy"].avg*100, self.fsb_accuracy_bsf*100))

        if self.model_params["dual_decoder"]:
            if self.rank==0: print(f'Rank {self.rank} >> Epoch {self.epoch}: Train ({100. * self.episode / self.trainer_params["train_episodes"]:.0f}%)  \n[Construction] Score: {self.metric_logger.construct_metrics["score"].avg:.4f},  Loss: {self.metric_logger.construct_metrics["loss"].avg:.4f} [{self.metric_logger.construct_metrics["loss1"].avg:.4f}, {self.metric_logger.construct_metrics["loss2"].avg:.4f}],  '
                  f'Infeasible_rate: [{self.metric_logger.construct_metrics["sol_infeasible_rate"].avg * 100:.4f}%, {self.metric_logger.construct_metrics["ins_infeasible_rate"].avg * 100:.4f}%], out: {self.metric_logger.construct_metrics["out"].avg:.4f}, out_nodes: {self.metric_logger.construct_metrics["out_nodes"].avg:.0f}, Feasible_dist: {self.metric_logger.construct_metrics["feasible_dist_max_pomo_mean"].avg:.4f}')

            if self.rank==0 and self.trainer_params["improve_steps"] > 0.: print(f'[Improvement] Score: {self.metric_logger.improve_metrics["current_score"].avg:.4f} [BSF: {self.metric_logger.improve_metrics["bsf_score"].avg:.4f}],  Loss: {self.metric_logger.improve_metrics["loss"].avg:.4f},  '
                  f'Infeasible_rate: [{self.metric_logger.improve_metrics["sol_infeasible_rate"].avg * 100:.4f}%, {self.metric_logger.improve_metrics["ins_infeasible_rate"].avg * 100:.4f}%], out: {self.metric_logger.improve_metrics["out"].avg:.4f}, out_nodes: {self.metric_logger.improve_metrics["out_nodes"].avg:.0f}, Feasible_dist: {self.metric_logger.improve_metrics["feasible_dist_max_pomo_mean"].avg:.4f}, Entropy: {self.metric_logger.improve_metrics["entropy"].avg:.4f}')
        else:
            if self.trainer_params["out_reward"]:
                if self.trainer_params["uncertainty_weight"]:
                    if self.rank == 0: print(
                        f'Rank {self.rank} >> Epoch {self.epoch}: Train ({100. * self.episode / self.trainer_params["train_episodes"]:.0f}%) \n[Construction] Score: {self.metric_logger.construct_metrics["score"].avg:.4f},  Loss: {self.metric_logger.construct_metrics["loss"].avg:.4f} ({self.metric_logger.sigma1.avg:.4f}),  Infeasible_rate: [{self.metric_logger.construct_metrics["sol_infeasible_rate"].avg * 100:.4f}%, {self.metric_logger.construct_metrics["ins_infeasible_rate"].avg * 100:.4f}%], '
                        f'out: {self.metric_logger.construct_metrics["out"].avg:.4f}, out_nodes: {self.metric_logger.construct_metrics["out_nodes"].avg:.0f}, Feasible_dist: {self.metric_logger.construct_metrics["feasible_dist_max_pomo_mean"].avg:.4f}')
                    if self.rank == 0 and self.trainer_params["improve_steps"] > 0.: print(
                        f'[Improvement] Score: {self.metric_logger.improve_metrics["current_score"].avg:.4f} [BSF: {self.metric_logger.improve_metrics["bsf_score"].avg:.4f}],  Loss: {self.metric_logger.improve_metrics["loss"].avg:.4f} ({self.metric_logger.sigma2.avg:.4f}),  Infeasible_rate: [{self.metric_logger.improve_metrics["sol_infeasible_rate"].avg * 100:.4f}%, {self.metric_logger.improve_metrics["ins_infeasible_rate"].avg * 100:.4f}%], '
                        f'out: {self.metric_logger.improve_metrics["out"].avg:.4f}, out_nodes: {self.metric_logger.improve_metrics["out_nodes"].avg:.0f}, Feasible_dist: {self.metric_logger.improve_metrics["feasible_dist_max_pomo_mean"].avg:.4f}, Entropy: {self.metric_logger.improve_metrics["entropy"].avg:.4f}')
                else:
                    if self.rank==0: print(f'Rank {self.rank} >> Epoch {self.epoch}: Train ({100. * self.episode / self.trainer_params["train_episodes"]:.0f}%) \n[Construction] Score: {self.metric_logger.construct_metrics["score"].avg:.4f},  Loss: {self.metric_logger.construct_metrics["loss"].avg:.4f},  Infeasible_rate: [{self.metric_logger.construct_metrics["sol_infeasible_rate"].avg * 100:.4f}%, {self.metric_logger.construct_metrics["ins_infeasible_rate"].avg * 100:.4f}%], '
                          f'out: {self.metric_logger.construct_metrics["out"].avg:.4f}, out_nodes: {self.metric_logger.construct_metrics["out_nodes"].avg:.0f}, Feasible_dist: {self.metric_logger.construct_metrics["feasible_dist_max_pomo_mean"].avg:.4f}')
                    if self.rank==0 and self.trainer_params["improve_steps"] > 0.: print(f'[Improvement] Score: {self.metric_logger.improve_metrics["current_score"].avg:.4f} [BSF: {self.metric_logger.improve_metrics["bsf_score"].avg:.4f}],  Loss: {self.metric_logger.improve_metrics["loss"].avg:.4f},  Infeasible_rate: [{self.metric_logger.improve_metrics["sol_infeasible_rate"].avg * 100:.4f}%, {self.metric_logger.improve_metrics["ins_infeasible_rate"].avg * 100:.4f}%], '
                          f'out: {self.metric_logger.improve_metrics["out"].avg:.4f}, out_nodes: {self.metric_logger.improve_metrics["out_nodes"].avg:.0f}, Feasible_dist: {self.metric_logger.improve_metrics["feasible_dist_max_pomo_mean"].avg:.4f}, Entropy: {self.metric_logger.improve_metrics["entropy"].avg:.4f}')
                    if self.rank==0 and self.trainer_params["improve_steps"] > 0.  and self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]: print(f'[Reconstruction] Score: {self.metric_logger.reconstruct_metrics["score"].avg:.4f},  Loss: {self.metric_logger.reconstruct_metrics["loss"].avg:.4f},  Infeasible_rate: [{self.metric_logger.reconstruct_metrics["sol_infeasible_rate"].avg * 100:.4f}%, {self.metric_logger.reconstruct_metrics["ins_infeasible_rate"].avg * 100:.4f}%], '
                          f'out: {self.metric_logger.reconstruct_metrics["out"].avg:.4f}, out_nodes: {self.metric_logger.reconstruct_metrics["out_nodes"].avg:.0f}, Feasible_dist: {self.metric_logger.reconstruct_metrics["feasible_dist_max_pomo_mean"].avg:.4f}')
            else:
                try:
                    if self.rank==0: print(f'Rank {self.rank} >> Epoch {self.epoch}: Train ({100. * self.episode / self.trainer_params["train_episodes"]:.0f}%) \n[Construction] Score: {self.metric_logger.construct_metrics["score"].avg:.4f},  Loss: {self.metric_logger.construct_metrics["loss"].avg:.4f}, '
                          f' Infeasible_rate: [{self.metric_logger.construct_metrics["sol_infeasible_rate"].avg * 100:.4f}%, {self.metric_logger.construct_metrics["ins_infeasible_rate"].avg * 100:.4f}%], Feasible_dist: {self.metric_logger.construct_metrics["feasible_dist_max_pomo_mean"].avg:.4f}')
                    if self.rank==0 and self.trainer_params["improve_steps"] > 0.: print(f'[Improvement] Score: {self.metric_logger.improve_metrics["current_score"].avg:.4f} [BSF: {self.metric_logger.improve_metrics["bsf_score"].avg:.4f}],  Loss: {self.metric_logger.improve_metrics["loss"].avg:.4f}, '
                          f' Infeasible_rate: [{self.metric_logger.improve_metrics["sol_infeasible_rate"].avg * 100:.4f}%, {self.metric_logger.improve_metrics["ins_infeasible_rate"].avg * 100:.4f}%], Feasible_dist: {self.metric_logger.improve_metrics["feasible_dist_max_pomo_mean"].avg:.4f}, Entropy: {self.metric_logger.improve_metrics["entropy"].avg:.4f}')
                except:
                    if self.rank==0: print(f'Rank {self.rank} >> Epoch {self.epoch}: Train ({100. * self.episode / self.trainer_params["train_episodes"]:.0f}%) \n[Construction] Score: {self.metric_logger.construct_metrics["score"].avg:.4f},  Loss: {self.metric_logger.construct_metrics["loss"].avg:.4f}')
                    if self.rank==0 and self.trainer_params["improve_steps"] > 0.: print(f'[Improvement] Score: {self.metric_logger.improve_metrics["current_score"].avg:.4f} [BSF: {self.metric_logger.improve_metrics["bsf_score"].avg:.4f}],  Loss: {self.metric_logger.improve_metrics["loss"].avg:.4f}, Entropy: {self.metric_logger.improve_metrics["entropy"].avg:.4f}')

        if self.rank == 0 and self.trainer_params["diversity_loss"]:
            print(f'Diversity Loss: {self.metric_logger.construct_metrics["diversity_loss"].avg:.4f}, Constructive RL loss: {self.metric_logger.construct_metrics["construct_RL_loss"].avg:.4f}')
            if self.trainer_params["improve_steps"] > 0. and self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]: print(f'[RC] Diversity Loss: {self.metric_logger.reconstruct_metrics["diversity_loss"].avg:.4f}, Constructive RL loss: {self.metric_logger.reconstruct_metrics["construct_RL_loss"].avg:.4f}')
        if self.rank == 0 and self.trainer_params["shared_critic"]:
            print(f'[construction] Actor Loss: {self.metric_logger.construct_metrics["construct_RL_loss"].avg:.4f}, Critic loss: {self.metric_logger.construct_metrics["critic_loss"].avg:.4f}, [improvement] Actor Loss: {self.metric_logger.improve_metrics["actor_loss"].avg:.4f}, Critic loss: {self.metric_logger.improve_metrics["critic_loss"].avg:.4f}')

    def _log_in_tb_logger(self, epoch): # train

        self.tb_logger.log_value('construction/train_score', self.metric_logger.construct_metrics["score"].avg, epoch)
        self.tb_logger.log_value('construction/train_loss', self.metric_logger.construct_metrics["loss"].avg, epoch)
        self.tb_logger.log_value('construction/construct_RL_loss', self.metric_logger.construct_metrics["construct_RL_loss"].avg, epoch)
        self.tb_logger.log_value('construction/diversity_loss', self.metric_logger.construct_metrics["diversity_loss"].avg, epoch)
        self.tb_logger.log_value('construction/critic_loss', self.metric_logger.construct_metrics["critic_loss"].avg, epoch)
        self.tb_logger.log_value('construction/max_vehicle_number', self.metric_logger.dummy_size.avg, epoch)
        self.tb_logger.log_value('construction/is_improved', self.metric_logger.construct_metrics["is_improved"].avg, epoch)
        self.tb_logger.log_value('construction/imitation_loss', self.metric_logger.construct_metrics["imitation_loss"].avg, epoch)
        self.tb_logger.log_value('coefficient', self.metric_logger.coefficient.avg, epoch)
        self.tb_logger.log_value('sigma1', self.metric_logger.sigma1.avg, epoch)
        self.tb_logger.log_value('sigma2', self.metric_logger.sigma2.avg, epoch)

        self.tb_logger.log_value('lambda_tw', self.metric_logger.lambda_tw.avg, epoch)
        self.tb_logger.log_value('lambda_demand', self.metric_logger.lambda_demand.avg, epoch)
        self.tb_logger.log_value('lambda_backhaul', self.metric_logger.lambda_backhaul.avg, epoch)
        self.tb_logger.log_value('lambda_dl', self.metric_logger.lambda_dl.avg, epoch)
        try:
            self.tb_logger.log_value('construction_feasibility/solution_infeasible_rate', self.metric_logger.construct_metrics["sol_infeasible_rate"].avg, epoch)
            self.tb_logger.log_value('construction_feasibility/instance_infeasible_rate', self.metric_logger.construct_metrics["ins_infeasible_rate"].avg, epoch)
        except:
            pass

        if self.model_params["pip_decoder"] and self.is_train_pip_decoder:
            self.tb_logger.log_value('sl_epoch/sl_loss', self.metric_logger.construct_metrics["sl_loss"].avg, epoch)
            self.tb_logger.log_value('sl_epoch/accuracy', self.metric_logger.construct_metrics["accuracy"].avg, epoch)
            self.tb_logger.log_value('sl_epoch/infsb_accuracy', self.metric_logger.construct_metrics["infsb_accuracy"].avg, epoch)
            self.tb_logger.log_value('sl_epoch/infsb_samples_number', self.metric_logger.construct_metrics["infsb_accuracy"].count, epoch)
            self.tb_logger.log_value('sl_epoch/fsb_accuracy', self.metric_logger.construct_metrics["fsb_accuracy"].avg, epoch)
            self.tb_logger.log_value('sl_epoch/fsb_samples_number', self.metric_logger.construct_metrics["fsb_accuracy"].count, epoch)

        if self.trainer_params["out_reward"]:
            self.tb_logger.log_value("construction_feasibility/total_out", self.metric_logger.construct_metrics["out"].avg, epoch)
            self.tb_logger.log_value("construction_feasibility/out_nodes", self.metric_logger.construct_metrics["out_nodes"].avg, epoch)
            if self.args.problem in ["OVRPBLTW", "OVRPLTW", "VRPBLTW"]:
                self.tb_logger.log_value("construction_feasibility/dlout", self.metric_logger.construct_metrics["dlout"].avg, epoch)
                self.tb_logger.log_value("construction_feasibility/dlout_nodes", self.metric_logger.construct_metrics["dlout_nodes"].avg, epoch)
                self.tb_logger.log_value("construction_feasibility/capacity_out", self.metric_logger.construct_metrics["capacity_out"].avg, epoch)
                self.tb_logger.log_value("construction_feasibility/capacity_out_nodes", self.metric_logger.construct_metrics["capacity_out_nodes"].avg, epoch)
        if self.trainer_params["fsb_dist_only"]:
            self.tb_logger.log_value("construction_feasibility/feasible_dist_mean", self.metric_logger.construct_metrics["feasible_dist_mean"].avg, epoch)
            self.tb_logger.log_value("construction_feasibility/feasible_dist_max_pomo_mean", self.metric_logger.construct_metrics["feasible_dist_max_pomo_mean"].avg, epoch)
        if self.model_params["dual_decoder"]:
            self.tb_logger.log_value('construction/train_loss_dist', self.metric_logger.construct_metrics["loss1"].avg, epoch)
            self.tb_logger.log_value('construction/train_loss_timeout', self.metric_logger.construct_metrics["loss2"].avg, epoch)

        if self.trainer_params["improve_steps"] >0:
            self.tb_logger.log_value('improvement/train_score_current', self.metric_logger.improve_metrics["current_score"].avg, epoch)
            self.tb_logger.log_value('improvement/train_score_bsf', self.metric_logger.improve_metrics["bsf_score"].avg, epoch)
            self.tb_logger.log_value('improvement/train_score_epsilon_fsb_bsf', self.metric_logger.improve_metrics["epsilon_fsb_bsf_score"].avg, epoch)
            self.tb_logger.log_value('improvement/train_improve_reward', self.metric_logger.improve_metrics["improve_reward"].avg, epoch)
            self.tb_logger.log_value('improvement/train_reg_reward', self.metric_logger.improve_metrics["reg_reward"].avg, epoch)
            self.tb_logger.log_value('improvement/train_bonus_reward', self.metric_logger.improve_metrics["bonus_reward"].avg, epoch)
            self.tb_logger.log_value('improvement/train_loss', self.metric_logger.improve_metrics["loss"].avg, epoch)
            self.tb_logger.log_value('improvement/actor_loss', self.metric_logger.improve_metrics["actor_loss"].avg, epoch)
            self.tb_logger.log_value('improvement/critic_loss', self.metric_logger.improve_metrics["critic_loss"].avg, epoch)
            self.tb_logger.log_value('improvement/entropy', self.metric_logger.improve_metrics["entropy"].avg, epoch)
            try:
                self.tb_logger.log_value('improvement_feasibility/solution_infeasible_rate', self.metric_logger.improve_metrics["sol_infeasible_rate"].avg , epoch)
                self.tb_logger.log_value('improvement_feasibility/instance_infeasible_rate', self.metric_logger.improve_metrics["ins_infeasible_rate"].avg, epoch)
                self.tb_logger.log_value('improvement_feasibility/epsilon_solution_infeasible_rate', self.metric_logger.improve_metrics["soft_sol_infeasible_rate"].avg, epoch)
                self.tb_logger.log_value('improvement_feasibility/epsilon_instance_infeasible_rate',self.metric_logger.improve_metrics["soft_ins_infeasible_rate"].avg, epoch)

            except:
                pass
            if self.trainer_params["out_reward"]:
                self.tb_logger.log_value("improvement_feasibility/total_out", self.metric_logger.improve_metrics["out"].avg, epoch)
                self.tb_logger.log_value("improvement_feasibility/out_nodes", self.metric_logger.improve_metrics["out_nodes"].avg, epoch)
                if self.args.problem in ["OVRPBLTW", "OVRPLTW", "VRPBLTW"]:
                    self.tb_logger.log_value("improvement_feasibility/tw_out", self.metric_logger.improve_metrics["tw_out"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility/tw_out_nodes", self.metric_logger.improve_metrics["tw_out_nodes"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility/capacity_out", self.metric_logger.improve_metrics["capacity_out"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility/capacity_out_nodes", self.metric_logger.improve_metrics["capacity_out_nodes"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility/dlout", self.metric_logger.improve_metrics["dlout"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility/dlout_nodes", self.metric_logger.improve_metrics["dlout_nodes"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility/backhaul_out", self.metric_logger.improve_metrics["backhaul_out"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility/backhaul_out_nodes", self.metric_logger.improve_metrics["backhaul_out_nodes"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility_ratio/tw_out_ratio", self.metric_logger.improve_metrics["tw_out_ratio"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility_ratio/capacity_out_ratio", self.metric_logger.improve_metrics["capacity_out_ratio"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility_ratio/backhaul_out_ratio", self.metric_logger.improve_metrics["backhaul_out_ratio"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility_ratio/dlout_ratio", self.metric_logger.improve_metrics["dlout_ratio"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility_ratio/out_ratio", self.metric_logger.improve_metrics["out_ratio"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility_ratio/cons_tw_out_ratio", self.metric_logger.improve_metrics["cons_tw_out_ratio"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility_ratio/cons_capacity_out_ratio", self.metric_logger.improve_metrics["cons_capacity_out_ratio"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility_ratio/cons_backhaul_out_ratio", self.metric_logger.improve_metrics["cons_backhaul_out_ratio"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility_ratio/cons_dlout_ratio", self.metric_logger.improve_metrics["cons_dlout_ratio"].avg, epoch)
                    self.tb_logger.log_value("improvement_feasibility_ratio/cons_out_ratio", self.metric_logger.improve_metrics["cons_out_ratio"].avg, epoch)
            if self.trainer_params["fsb_dist_only"]:
                self.tb_logger.log_value("improvement_feasibility/feasible_dist_mean", self.metric_logger.improve_metrics["feasible_dist_mean"].avg, epoch)
                self.tb_logger.log_value("improvement_feasibility/feasible_dist_max_pomo_mean", self.metric_logger.improve_metrics["feasible_dist_max_pomo_mean"].avg, epoch)
                self.tb_logger.log_value("improvement_feasibility/epsilon_feasible_dist_mean", self.metric_logger.improve_metrics["epsilon_feasible_dist_mean"].avg, epoch)
                self.tb_logger.log_value("improvement_feasibility/epsilon_feasible_dist_max_pomo_mean", self.metric_logger.improve_metrics["epsilon_feasible_dist_max_pomo_mean"].avg, epoch)
            # if self.model_params["dual_decoder"]:
            #     self.tb_logger.log_value('construction/train_loss_dist', self.metric_logger.improve_metrics["loss1"].avg, epoch)
            #     self.tb_logger.log_value('construction/train_loss_timeout', self.metric_logger.improve_metrics["loss2"].avg, epoch)
            if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]:
                self.tb_logger.log_value('reconstruction/train_score', self.metric_logger.reconstruct_metrics["score"].avg,epoch)
                self.tb_logger.log_value('reconstruction/train_loss', self.metric_logger.reconstruct_metrics["loss"].avg,epoch)
                self.tb_logger.log_value('reconstruction/construct_RL_loss',self.metric_logger.reconstruct_metrics["construct_RL_loss"].avg, epoch)
                self.tb_logger.log_value('reconstruction/diversity_loss',self.metric_logger.reconstruct_metrics["diversity_loss"].avg, epoch)
                try:
                    self.tb_logger.log_value('reconstruction_feasibility/solution_infeasible_rate',self.metric_logger.reconstruct_metrics["sol_infeasible_rate"].avg, epoch)
                    self.tb_logger.log_value('reconstruction_feasibility/instance_infeasible_rate',self.metric_logger.reconstruct_metrics["ins_infeasible_rate"].avg, epoch)
                except:
                    pass
                if self.trainer_params["out_reward"]:
                    self.tb_logger.log_value("reconstruction_feasibility/total_out",self.metric_logger.reconstruct_metrics["out"].avg, epoch)
                    self.tb_logger.log_value("reconstruction_feasibility/out_nodes", self.metric_logger.reconstruct_metrics["out_nodes"].avg, epoch)
                    if self.args.problem in ["OVRPBLTW", "OVRPLTW", "VRPBLTW"]:
                        self.tb_logger.log_value("reconstruction_feasibility/dlout",self.metric_logger.reconstruct_metrics["dlout"].avg, epoch)
                        self.tb_logger.log_value("reconstruction_feasibility/dlout_nodes",self.metric_logger.reconstruct_metrics["dlout_nodes"].avg, epoch)
                        self.tb_logger.log_value("reconstruction_feasibility/capacity_out", self.metric_logger.reconstruct_metrics["capacity_out"].avg, epoch)
                        self.tb_logger.log_value("reconstruction_feasibility/capacity_out_nodes",self.metric_logger.reconstruct_metrics["capacity_out_nodes"].avg, epoch)
                if self.trainer_params["fsb_dist_only"]:
                    self.tb_logger.log_value("reconstruction_feasibility/feasible_dist_mean",self.metric_logger.reconstruct_metrics["feasible_dist_mean"].avg, epoch)
                    self.tb_logger.log_value("reconstruction_feasibility/feasible_dist_max_pomo_mean",self.metric_logger.reconstruct_metrics["feasible_dist_max_pomo_mean"].avg,epoch)

    def _log_in_wandb(self, epoch): # train
        log_data = {
            'construction/train_score': self.metric_logger.construct_metrics["score"].avg,
            'construction/train_loss': self.metric_logger.construct_metrics["loss"].avg,
            'construction/construct_RL_loss': self.metric_logger.construct_metrics["construct_RL_loss"].avg,
            'construction/diversity_loss': self.metric_logger.construct_metrics["diversity_loss"].avg,
            'construction/critic_loss': self.metric_logger.construct_metrics["critic_loss"].avg,
            'construction/is_improved': self.metric_logger.construct_metrics["is_improved"].avg,
            'construction/imitation_loss': self.metric_logger.construct_metrics["imitation_loss"].avg,
            'construction/max_vehicle_number': self.metric_logger.dummy_size.avg,
            'coefficient': self.metric_logger.coefficient.avg,
            'sigma1': self.metric_logger.sigma1.avg,
            'sigma2': self.metric_logger.sigma2.avg,
            'lambda_tw': self.metric_logger.lambda_tw.avg,
            'lambda_demand': self.metric_logger.lambda_demand.avg,
            'lambda_backhaul': self.metric_logger.lambda_backhaul.avg,
            'lambda_dl': self.metric_logger.lambda_dl.avg,
            'construction_feasibility/solution_infeasible_rate': self.metric_logger.construct_metrics["sol_infeasible_rate"].avg,
            'construction_feasibility/instance_infeasible_rate': self.metric_logger.construct_metrics["ins_infeasible_rate"].avg
        }

        if self.model_params["pip_decoder"] and self.is_train_pip_decoder:
            log_data.update({
                'sl_epoch/sl_loss': self.metric_logger.construct_metrics["sl_loss"].avg,
                'sl_epoch/accuracy': self.metric_logger.construct_metrics["accuracy"].avg,
                'sl_epoch/infsb_accuracy': self.metric_logger.construct_metrics["infsb_accuracy"].avg,
                'sl_epoch/infsb_samples_number': self.metric_logger.construct_metrics["infsb_accuracy"].count,
                'sl_epoch/fsb_accuracy': self.metric_logger.construct_metrics["fsb_accuracy"].avg,
                'sl_epoch/fsb_samples_number': self.metric_logger.construct_metrics["fsb_accuracy"].count,
            })

        if self.trainer_params["out_reward"]:
            log_data.update({
                'construction_feasibility/total_out': self.metric_logger.construct_metrics["out"].avg,
                'construction_feasibility/out_nodes': self.metric_logger.construct_metrics["out_nodes"].avg,
            })
            if self.args.problem in ["OVRPBLTW", "OVRPLTW", "VRPBLTW"]:
                log_data.update({
                    'construction_feasibility/dlout': self.metric_logger.construct_metrics["dlout"].avg,
                    'construction_feasibility/dlout_nodes': self.metric_logger.construct_metrics["dlout_nodes"].avg,
                    'construction_feasibility/capacity_out': self.metric_logger.construct_metrics["capacity_out"].avg,
                    'construction_feasibility/capacity_out_nodes': self.metric_logger.construct_metrics["capacity_out_nodes"].avg,
                })

        if self.trainer_params["fsb_dist_only"]:
            log_data.update({
                'construction_feasibility/feasible_dist_mean': self.metric_logger.construct_metrics[
                    "feasible_dist_mean"].avg,
                'construction_feasibility/feasible_dist_max_pomo_mean': self.metric_logger.construct_metrics[
                    "feasible_dist_max_pomo_mean"].avg,
            })

        if self.model_params["dual_decoder"]:
            log_data.update({
                'construction/train_loss_dist': self.metric_logger.construct_metrics["loss1"].avg,
                'construction/train_loss_timeout': self.metric_logger.construct_metrics["loss2"].avg,
            })

        if self.trainer_params["improve_steps"] > 0:
            log_data.update({
                'improvement/train_score_current': self.metric_logger.improve_metrics["current_score"].avg,
                'improvement/train_score_bsf': self.metric_logger.improve_metrics["bsf_score"].avg,
                'improvement/train_score_epsilon_fsb_bsf': self.metric_logger.improve_metrics["epsilon_fsb_bsf_score"].avg,
                'improvement/train_improve_reward': self.metric_logger.improve_metrics["improve_reward"].avg,
                'improvement/train_reg_reward': self.metric_logger.improve_metrics["reg_reward"].avg,
                'improvement/train_bonus_reward': self.metric_logger.improve_metrics["bonus_reward"].avg,
                'improvement/actor_loss': self.metric_logger.improve_metrics["loss"].avg,
                'improvement/critic_loss': self.metric_logger.improve_metrics["actor_loss"].avg,
                'improvement/critic_loss': self.metric_logger.improve_metrics["critic_loss"].avg,
                'improvement/entropy': self.metric_logger.improve_metrics["entropy"].avg,
                'improvement_feasibility/solution_infeasible_rate': self.metric_logger.improve_metrics["sol_infeasible_rate"].avg,
                'improvement_feasibility/instance_infeasible_rate': self.metric_logger.improve_metrics["ins_infeasible_rate"].avg,
                'improvement_feasibility/epsilon_solution_infeasible_rate': self.metric_logger.improve_metrics["soft_sol_infeasible_rate"].avg,
                'improvement_feasibility/epsilon_instance_infeasible_rate': self.metric_logger.improve_metrics["soft_ins_infeasible_rate"].avg,
            })

            if self.trainer_params["out_reward"]:
                log_data.update({
                    'improvement_feasibility/total_out': self.metric_logger.improve_metrics["out"].avg,
                    'improvement_feasibility/out_nodes': self.metric_logger.improve_metrics["out_nodes"].avg,
                })
                if self.args.problem in ["VRPBLTW"]:
                    log_data.update({
                        'improvement_feasibility/tw_out': self.metric_logger.improve_metrics["tw_out"].avg,
                        'improvement_feasibility/tw_out_nodes': self.metric_logger.improve_metrics["tw_out_nodes"].avg,
                        'improvement_feasibility/capacity_out': self.metric_logger.improve_metrics["capacity_out"].avg,
                        'improvement_feasibility/capacity_out_nodes': self.metric_logger.improve_metrics["capacity_out_nodes"].avg,
                        'improvement_feasibility/dlout': self.metric_logger.improve_metrics["dlout"].avg,
                        'improvement_feasibility/dlout_nodes': self.metric_logger.improve_metrics["dlout_nodes"].avg,
                        'improvement_feasibility/backhaul_out': self.metric_logger.improve_metrics["backhaul_out"].avg,
                        'improvement_feasibility/backhaul_out_nodes': self.metric_logger.improve_metrics["backhaul_out_nodes"].avg,
                        'improvement_feasibility_ratio/tw_out_ratio': self.metric_logger.improve_metrics["tw_out_ratio"].avg,
                        'improvement_feasibility_ratio/capacity_out_ratio': self.metric_logger.improve_metrics["capacity_out_ratio"].avg,
                        'improvement_feasibility_ratio/backhaul_out_ratio': self.metric_logger.improve_metrics["backhaul_out_ratio"].avg,
                        'improvement_feasibility_ratio/dlout_ratio': self.metric_logger.improve_metrics["dlout_ratio"].avg,
                        'improvement_feasibility_ratio/out_ratio': self.metric_logger.improve_metrics["out_ratio"].avg,
                        'improvement_feasibility_ratio/cons_tw_out_ratio': self.metric_logger.improve_metrics["cons_tw_out_ratio"].avg,
                        'improvement_feasibility_ratio/cons_capacity_out_ratio': self.metric_logger.improve_metrics["cons_capacity_out_ratio"].avg,
                        'improvement_feasibility_ratio/cons_backhaul_out_ratio': self.metric_logger.improve_metrics["cons_backhaul_out_ratio"].avg,
                        'improvement_feasibility_ratio/cons_dlout_ratio': self.metric_logger.improve_metrics["cons_dlout_ratio"].avg,
                        'improvement_feasibility_ratio/cons_out_ratio': self.metric_logger.improve_metrics["cons_out_ratio"].avg,
                    })

            if self.trainer_params["fsb_dist_only"]:
                log_data.update({
                    'improvement_feasibility/feasible_dist_mean': self.metric_logger.improve_metrics["feasible_dist_mean"].avg,
                    'improvement_feasibility/feasible_dist_max_pomo_mean': self.metric_logger.improve_metrics["feasible_dist_max_pomo_mean"].avg,
                    'improvement_feasibility/epsilon_feasible_dist_mean': self.metric_logger.improve_metrics["epsilon_feasible_dist_mean"].avg,
                    'improvement_feasibility/epsilon_feasible_dist_max_pomo_mean': self.metric_logger.improve_metrics["epsilon_feasible_dist_max_pomo_mean"].avg,
                })

            if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]:
                log_data.update({
                    'reconstruction/train_score': self.metric_logger.reconstruct_metrics["score"].avg,
                    'reconstruction/train_loss': self.metric_logger.reconstruct_metrics["loss"].avg,
                    'reconstruction/reconstruct_RL_loss': self.metric_logger.reconstruct_metrics["construct_RL_loss"].avg,
                    'reconstruction/diversity_loss': self.metric_logger.reconstruct_metrics["diversity_loss"].avg,
                    'reconstruction_feasibility/solution_infeasible_rate': self.metric_logger.reconstruct_metrics["sol_infeasible_rate"].avg,
                    'reconstruction_feasibility/instance_infeasible_rate': self.metric_logger.reconstruct_metrics["ins_infeasible_rate"].avg
                })

                if self.trainer_params["out_reward"]:
                    log_data.update({
                        'reconstruction_feasibility/total_out': self.metric_logger.reconstruct_metrics["out"].avg,
                        'reconstruction_feasibility/out_nodes': self.metric_logger.reconstruct_metrics["out_nodes"].avg,
                    })
                    if self.args.problem in ["OVRPBLTW", "OVRPLTW", "VRPBLTW"]:
                        log_data.update({
                            'reconstruction_feasibility/dlout': self.metric_logger.reconstruct_metrics["dlout"].avg,
                            'reconstruction_feasibility/dlout_nodes': self.metric_logger.reconstruct_metrics["dlout_nodes"].avg,
                            'reconstruction_feasibility/capacity_out': self.metric_logger.reconstruct_metrics["capacity_out"].avg,
                            'reconstruction_feasibility/capacity_out_nodes': self.metric_logger.reconstruct_metrics["capacity_out_nodes"].avg,
                        })

                if self.trainer_params["fsb_dist_only"]:
                    log_data.update({
                        'reconstruction_feasibility/feasible_dist_mean': self.metric_logger.reconstruct_metrics["feasible_dist_mean"].avg,
                        'reconstruction_feasibility/feasible_dist_max_pomo_mean': self.metric_logger.reconstruct_metrics["feasible_dist_max_pomo_mean"].avg,
                    })

        wandb.log(log_data, step=epoch)

    def _save_best_model(self, score_cons, score_impr, mode="train"):
        try:
            if not self.trainer_params["improvement_only"]:
                if score_cons < self.best_score_cons[mode]:
                    self.best_score_cons[mode] = score_cons
                    if self.rank == 0:
                        torch.save(self.model.state_dict(), os.path.join(self.log_path, f'{mode}_model_best_cons.pt'))
                        print(f"Rank {self.rank} >> Best model of construction saved!")
            if self.trainer_params["improve_steps"] > 0.:
                if score_impr < self.best_score_impr[mode]:
                    self.best_score_impr[mode] = score_impr
                    if self.rank == 0:
                        torch.save(self.model.state_dict(), os.path.join(self.log_path, f'{mode}_model_best_impr.pt'))
                        print(f"Rank {self.rank} >> Best model of improvement saved!")
        except:
            self.best_score_cons[mode] = score_cons
            self.best_score_impr[mode] = score_impr

    def  _val_logger(self, epoch):

        score = torch.tensor([0.]).cuda()
        gap = torch.tensor([0.]).cuda()
        sol_infeasible_rate_list = torch.tensor([0.]).cuda()
        ins_infeasible_rate_list = torch.tensor([0.]).cuda()

        if self.tester_params["aux_mask"]:
            rc_masked_score = torch.tensor([0.]).cuda()
            rc_masked_gap = torch.tensor([0.]).cuda()
            rc_masked_sol_infeasible_rate_list = torch.tensor([0.]).cuda()
            rc_masked_ins_infeasible_rate_list = torch.tensor([0.]).cuda()

        if self.trainer_params["validation_improve_steps"] > 0.:
            improve_score = torch.tensor([0.]).cuda()
            improve_gap = torch.tensor([0.]).cuda()
            improve_sol_infeasible_rate_list = torch.tensor([0.]).cuda()
            improve_ins_infeasible_rate_list = torch.tensor([0.]).cuda()
            if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]:
                rc_score = torch.tensor([0.]).cuda()
                rc_gap = torch.tensor([0.]).cuda()
                rc_sol_infeasible_rate_list = torch.tensor([0.]).cuda()
                rc_ins_infeasible_rate_list = torch.tensor([0.]).cuda()

        if not self.trainer_params["improvement_only"]:
            score = self.val_metric_logger.construct_metrics["aug_score_list"]
            gap = self.val_metric_logger.construct_metrics["aug_gap_list"]
            sol_infeasible_rate_list = self.val_metric_logger.construct_metrics["sol_infeasible_rate_list"]
            ins_infeasible_rate_list = self.val_metric_logger.construct_metrics["ins_infeasible_rate_list"]

        if self.tester_params["aux_mask"]:
            rc_masked_score = self.val_metric_logger.reconstruct_masked_metrics["aug_score_list"]
            rc_masked_gap = self.val_metric_logger.reconstruct_masked_metrics["aug_gap_list"]
            rc_masked_sol_infeasible_rate_list = self.val_metric_logger.reconstruct_masked_metrics["sol_infeasible_rate_list"]
            rc_masked_ins_infeasible_rate_list = self.val_metric_logger.reconstruct_masked_metrics["ins_infeasible_rate_list"]

        if self.trainer_params["validation_improve_steps"] > 0.:
            improve_score = self.val_metric_logger.improve_metrics["aug_score_list"]
            improve_gap = self.val_metric_logger.improve_metrics["aug_gap_list"]
            improve_sol_infeasible_rate_list = self.val_metric_logger.improve_metrics["sol_infeasible_rate_list"]
            improve_ins_infeasible_rate_list = self.val_metric_logger.improve_metrics["ins_infeasible_rate_list"]
            if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]:
                rc_score = self.val_metric_logger.reconstruct_masked_metrics["aug_score_list"]
                rc_gap = self.val_metric_logger.reconstruct_masked_metrics["aug_gap_list"]
                rc_sol_infeasible_rate_list = self.val_metric_logger.reconstruct_masked_metrics["sol_infeasible_rate_list"]
                rc_ins_infeasible_rate_list = self.val_metric_logger.reconstruct_masked_metrics["ins_infeasible_rate_list"]

        if self.trainer_params["pip_decoder"]:
            accuracy = self.val_metric_logger.construct_metrics["accuracy"]
            infsb_accuracy = self.val_metric_logger.construct_metrics["infsb_accuracy"]
            fsb_accuracy = self.val_metric_logger.construct_metrics["fsb_accuracy"]
            infsb_sample_nums = self.val_metric_logger.construct_metrics["infsb_sample_nums"]
            fsb_sample_nums = self.val_metric_logger.construct_metrics["fsb_sample_nums"]

        if self.args.multiple_gpu:
            dist.barrier()
            if not self.trainer_params["improvement_only"]:
                score = gather_tensor_and_concat(torch.tensor([score]).cuda())
                gap = gather_tensor_and_concat(torch.tensor([gap]).cuda())
                sol_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([sol_infeasible_rate_list]).cuda())
                ins_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([ins_infeasible_rate_list]).cuda())
            if self.tester_params["aux_mask"]:
                rc_masked_score = gather_tensor_and_concat(torch.tensor([rc_masked_score]).cuda())
                rc_masked_gap = gather_tensor_and_concat(torch.tensor([rc_masked_gap]).cuda())
                rc_masked_sol_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([rc_masked_sol_infeasible_rate_list]).cuda())
                rc_masked_ins_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([rc_masked_ins_infeasible_rate_list]).cuda())
            if self.trainer_params["validation_improve_steps"] > 0.:
                improve_score = gather_tensor_and_concat(torch.tensor([improve_score]).cuda())
                improve_gap = gather_tensor_and_concat(torch.tensor([improve_gap]).cuda())
                improve_sol_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([improve_sol_infeasible_rate_list]).cuda())
                improve_ins_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([improve_ins_infeasible_rate_list]).cuda())
                if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]:
                    rc_score = gather_tensor_and_concat(torch.tensor([rc_score]).cuda())
                    rc_gap = gather_tensor_and_concat(torch.tensor([rc_gap]).cuda())
                    rc_sol_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([rc_sol_infeasible_rate_list]).cuda())
                    rc_ins_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([rc_ins_infeasible_rate_list]).cuda())
            if self.trainer_params["pip_decoder"]:
                accuracy = gather_tensor_and_concat(torch.tensor([accuracy]).cuda())
                infsb_accuracy = gather_tensor_and_concat(torch.tensor([infsb_accuracy]).cuda())
                fsb_accuracy = gather_tensor_and_concat(torch.tensor([fsb_accuracy]).cuda())
                infsb_sample_nums = gather_tensor_and_concat(torch.tensor([infsb_sample_nums]).cuda())
                fsb_sample_nums = gather_tensor_and_concat(torch.tensor([fsb_sample_nums]).cuda())
            dist.barrier()
            score = score.mean()
            gap = gap.mean()
            ins_infeasible_rate_list = ins_infeasible_rate_list.mean()
            sol_infeasible_rate_list = sol_infeasible_rate_list.mean()
            if self.tester_params["aux_mask"]:
                rc_masked_score = rc_masked_score.mean()
                rc_masked_gap = rc_masked_gap.mean()
                rc_masked_ins_infeasible_rate_list = rc_masked_ins_infeasible_rate_list.mean()
                rc_masked_sol_infeasible_rate_list = rc_masked_sol_infeasible_rate_list.mean()
            if self.trainer_params["validation_improve_steps"] > 0.:
                improve_score = improve_score.mean()
                improve_gap = improve_gap.mean()
                improve_sol_infeasible_rate_list = improve_sol_infeasible_rate_list.mean()
                improve_ins_infeasible_rate_list = improve_ins_infeasible_rate_list.mean()
                if self.trainer_params["reconstruct"]:
                    rc_score = rc_score.mean()
                    rc_gap = rc_gap.mean()
                    rc_ins_infeasible_rate_list = rc_ins_infeasible_rate_list.mean()
                    rc_sol_infeasible_rate_list = rc_sol_infeasible_rate_list.mean()
            if self.trainer_params["pip_decoder"]:
                accuracy = accuracy.mean()
                infsb_accuracy = infsb_accuracy.mean()
                fsb_accuracy = fsb_accuracy.mean()
                infsb_sample_nums = infsb_sample_nums.sum()
                fsb_sample_nums = fsb_sample_nums.sum()

        if self.rank == 0:
            if not self.trainer_params["improvement_only"]:
                self.result_log["val_score"].append(score)
                self.result_log["val_gap"].append(gap)
                self.result_log["val_infsb_rate"].append([sol_infeasible_rate_list, ins_infeasible_rate_list])
            if self.tester_params["aux_mask"]:
                self.result_log["rc_masked_val_score"].append(rc_masked_score)
                self.result_log["rc_masked_val_gap"].append(rc_masked_gap)
                self.result_log["rc_masked_val_infsb_rate"].append([rc_masked_sol_infeasible_rate_list, rc_masked_ins_infeasible_rate_list])
            if self.trainer_params["validation_improve_steps"] > 0.:
                self.result_log["val_score_improve"].append(improve_score)
                self.result_log["val_gap_improve"].append(improve_gap)
                self.result_log["val_infsb_rate_improve"].append([improve_sol_infeasible_rate_list, improve_ins_infeasible_rate_list])
                if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]:
                    self.result_log["rc_val_score"].append(rc_score)
                    self.result_log["rc_val_gap"].append(rc_gap)
                    self.result_log["rc_val_infsb_rate"].append([rc_sol_infeasible_rate_list, rc_ins_infeasible_rate_list])

        if self.args.wandb_logger and self.rank == 0:
            logdata = {}
            if not self.trainer_params["improvement_only"]:
                logdata ={"val/val_score": score,
                            "val/val_gap": gap,
                            'val/val_sol_infsb_rate': sol_infeasible_rate_list,
                            'val/val_ins_infsb_rate': ins_infeasible_rate_list}
            if self.tester_params["aux_mask"]:
                logdata.update({"val/val_score_rc_masked": rc_masked_score,
                                "val/val_gap_rc_masked": rc_masked_gap,
                                'val/val_sol_infsb_rate_rc_masked': rc_masked_sol_infeasible_rate_list,
                                'val/val_ins_infsb_rate_rc_masked': rc_masked_ins_infeasible_rate_list})
            if self.trainer_params["validation_improve_steps"] > 0.:
                logdata.update({"val/improve_val_score": improve_score,
                                "val/improve_val_gap": improve_gap,
                                'val/improve_val_sol_infsb_rate': improve_sol_infeasible_rate_list,
                                'val/improve_val_ins_infsb_rate': improve_ins_infeasible_rate_list})
                if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]:
                    logdata.update({"val/rc_val_score": rc_score,
                                    "val/rc_val_gap": rc_gap,
                                    'val/rc_val_sol_infsb_rate': rc_sol_infeasible_rate_list,
                                    'val/rc_val_ins_infsb_rate': rc_ins_infeasible_rate_list})
            if self.trainer_params["pip_decoder"]:
                logdata.update({'val_pipd/accuracy': accuracy,
                               'val_pipd/infsb_accuracy': infsb_accuracy,
                               'val_pipd/fsb_accuracy': fsb_accuracy,
                               'val_pipd/infsb_sample_nums': infsb_sample_nums,
                               'val_pipd/fsb_sample_nums': fsb_sample_nums})
            wandb.log(logdata, step = epoch)

        if self.tb_logger and self.rank == 0:
            if not self.trainer_params["improvement_only"]:
                self.tb_logger.log_value('val/val_score', score, epoch)
                self.tb_logger.log_value('val/val_gap', gap, epoch)
            if self.tester_params["aux_mask"]:
                self.tb_logger.log_value('val/val_score_rc_masked', rc_masked_score, epoch)
                self.tb_logger.log_value('val/val_gap_rc_masked', rc_masked_gap, epoch)
            if self.trainer_params["validation_improve_steps"] > 0.:
                self.tb_logger.log_value('val/improve_val_score', improve_score, epoch)
                self.tb_logger.log_value('val/improve_val_gap', improve_gap, epoch)
                if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]:
                    self.tb_logger.log_value('val/val_score_rc', rc_score, epoch)
                    self.tb_logger.log_value('val/val_gap_rc', rc_gap, epoch)
            try:
                if not self.trainer_params["improvement_only"]:
                    self.tb_logger.log_value('val/val_sol_infsb_rate', sol_infeasible_rate_list, epoch)
                    self.tb_logger.log_value('val/val_ins_infsb_rate', ins_infeasible_rate_list, epoch)
                if self.tester_params["aux_mask"]:
                    self.tb_logger.log_value('val/val_sol_infsb_rate_rc_masked', rc_masked_sol_infeasible_rate_list, epoch)
                    self.tb_logger.log_value('val/val_ins_infsb_rate_rc_masked', rc_masked_ins_infeasible_rate_list, epoch)
                if self.trainer_params["validation_improve_steps"] > 0.:
                    self.tb_logger.log_value('val/improve_val_sol_infsb_rate', improve_sol_infeasible_rate_list, epoch)
                    self.tb_logger.log_value('val/improve_val_ins_infsb_rate', improve_ins_infeasible_rate_list, epoch)
                    if self.args.problem == "VRPBLTW" and self.trainer_params["reconstruct"]:
                        self.tb_logger.log_value('val/val_sol_infsb_rate_rc', rc_sol_infeasible_rate_list, epoch)
                        self.tb_logger.log_value('val/val_ins_infsb_rate_rc', rc_ins_infeasible_rate_list, epoch)
            except:
                pass
            if self.trainer_params["pip_decoder"]:
                self.tb_logger.log_value('val_pipd/accuracy', accuracy, epoch)
                self.tb_logger.log_value('val_pipd/infsb_accuracy', infsb_accuracy, epoch)
                self.tb_logger.log_value('val_pipd/fsb_accuracy', fsb_accuracy, epoch)
                self.tb_logger.log_value('val_pipd/infsb_sample_nums', infsb_sample_nums,epoch)
                self.tb_logger.log_value('val_pipd/fsb_sample_nums', fsb_sample_nums, epoch)

    def sample_z_vectors(self, batch_size, starting_points, z_dim, z_sample_size, rollout_size):

        if 2**z_dim == rollout_size:
            z = self.binary_string_pool[None].expand(batch_size, rollout_size, z_dim)
        else:
            z_idx = torch.multinomial((torch.ones(batch_size * starting_points, 2**z_dim) / 2**z_dim),
                                  z_sample_size, replacement=z_sample_size > 2**z_dim)
            z = self.binary_string_pool[z_idx].reshape(batch_size, starting_points, z_sample_size, z_dim)
            z = z.transpose(1, 2).reshape(batch_size, rollout_size, z_dim)
        return z

    def _get_pomo_initial_solution(self, env, test_data, batch_size, rollout_size, eval_type, aug_factor, val=False):
        assert self.pomo_model.training == False, ">>>>>>>>> ERROR"
        self.pomo_model.eval()
        try:
            self.pomo_model.module.set_eval_type(eval_type)
        except:
            self.pomo_model.set_eval_type(eval_type)
        with torch.no_grad():
            env.load_problems(batch_size, rollout_size, problems=test_data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            self.pomo_model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _, _ = self.pomo_model(state, pomo=self.env_params["pomo_start"],
                                            candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None)
                # shape: (batch, pomo)
                use_predicted_PI_mask=False
                state, reward, done, infeasible = env.step(selected,
                                                           out_reward=self.trainer_params["out_reward"] if not val else False,
                                                           soft_constrained=self.trainer_params["soft_constrained"],
                                                           backhaul_mask=self.trainer_params["backhaul_mask"],
                                                           generate_PI_mask=self.trainer_params["generate_PI_mask"],
                                                           use_predicted_PI_mask=use_predicted_PI_mask,
                                                           pip_step=self.trainer_params["pip_step"]
                                                           )

        if not val:
            self._get_construction_output(infeasible, reward)
            return torch.stack(reward).sum(0)
        else:
            self._get_construction_output_val(aug_factor, infeasible, reward, env.selected_node_list)

    def _save_pip_decoder(self, accuracy, infsb_accuracy, fsb_accuracy):
        # save lazy model every epoch
        # FIXME: only consider rank 1 if using multiple GPUs
        accuracy_AM_avg = self.metric_logger.construct_metrics["accuracy"].avg
        fsb_accuracy_AM_avg = self.metric_logger.construct_metrics["fsb_accuracy"].avg
        infsb_accuracy_AM_avg = self.metric_logger.construct_metrics["infsb_accuracy"].avg

        self.accuracy_isbsf = True if accuracy_AM_avg > self.accuracy_bsf else False
        self.fsb_accuracy_isbsf = True if fsb_accuracy_AM_avg > self.fsb_accuracy_bsf else False
        self.infsb_accuracy_isbsf = True if infsb_accuracy_AM_avg > self.infsb_accuracy_bsf else False

        self.accuracy_bsf = accuracy_AM_avg if accuracy_AM_avg > self.accuracy_bsf else self.accuracy_bsf
        self.fsb_accuracy_bsf = fsb_accuracy_AM_avg if fsb_accuracy_AM_avg > self.fsb_accuracy_bsf else self.fsb_accuracy_bsf
        self.infsb_accuracy_bsf = infsb_accuracy_AM_avg if infsb_accuracy_AM_avg > self.infsb_accuracy_bsf else self.infsb_accuracy_bsf

        if self.accuracy_isbsf:
            if not os.path.exists('{}/accuracy_bsf.pt'.format(self.log_path)) or (infsb_accuracy > 0.75 and fsb_accuracy > 0.75):
                # if not exist, save
                # then check whether the current batch is bad, if no then save
                print(
                    "Saving BSF accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(
                        self.accuracy_bsf * 100, accuracy * 100, infsb_accuracy * 100, fsb_accuracy * 100))
                checkpoint_dict = {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': accuracy_AM_avg,
                    'fsb_accuracy': fsb_accuracy_AM_avg,
                    'infsb_accuracy': infsb_accuracy_AM_avg,
                }
                torch.save(checkpoint_dict, '{}/accuracy_bsf.pt'.format(self.log_path))
        if self.fsb_accuracy_isbsf:
            if not os.path.exists('{}/fsb_accuracy_bsf.pt'.format(self.log_path)) or infsb_accuracy > 0.75 or (infsb_accuracy > 0.6 and self.problem == "TSPDL"):
                # if not exist, save
                # then check whether the current batch is bad, if yes then don't save
                print(
                    "Saving BSF Feasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(
                        self.fsb_accuracy_bsf * 100, accuracy * 100, infsb_accuracy * 100, fsb_accuracy * 100))
                checkpoint_dict = {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': accuracy_AM_avg,
                    'fsb_accuracy': fsb_accuracy_AM_avg,
                    'infsb_accuracy': infsb_accuracy_AM_avg,
                }
                torch.save(checkpoint_dict, '{}/fsb_accuracy_bsf.pt'.format(self.log_path))
        if self.infsb_accuracy_isbsf:
            if not os.path.exists('{}/infsb_accuracy_bsf.pt'.format(self.log_path)) or fsb_accuracy > 0.75 or (fsb_accuracy > 0.6 and self.problem == "TSPDL"):
                # if not exist, save
                # then check whether the current batch is bad, if yes then don't save
                print(
                    "Saving BSF Infeasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(
                        self.infsb_accuracy_bsf * 100, accuracy * 100, infsb_accuracy * 100, fsb_accuracy * 100))
                checkpoint_dict = {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': accuracy_AM_avg,
                    'fsb_accuracy': fsb_accuracy_AM_avg,
                    'infsb_accuracy': infsb_accuracy_AM_avg,
                }
                torch.save(checkpoint_dict, '{}/infsb_accuracy_bsf.pt'.format(self.log_path))