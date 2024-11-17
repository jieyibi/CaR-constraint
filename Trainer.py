import copy
import re
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from tensorboard_logger import Logger as TbLogger
from utils import *
# from sklearn.utils.class_weight import compute_class_weight
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from torch.utils.data import DataLoader, DistributedSampler  # use pytorch dataloader
# from sklearn.metrics import confusion_matrix, roc_auc_score
import csv
from datetime import datetime
import pytz
import pdb
import wandb
import itertools
import torch
from collections import OrderedDict


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
        self.log_path = args.log_path if rank == 0 else None
        self.result_log = {"val_score": [], "val_gap": [], "val_infsb_rate": []}
        if self.trainer_params["validation_improve_steps"] > 0.:
            self.result_log.update({"val_score_improve": [], "val_gap_improve": [], "val_infsb_rate_improve": []})
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
            self.lambda_ = torch.ones((trainer_params["constraint_number"],), requires_grad=False) if trainer_params["adaptive_primal_dual"] else self.trainer_params["penalty_factor"]
            self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.scaler = torch.cuda.amp.GradScaler()
        num_param(self.model)

        self.penalty_factor = self.trainer_params["penalty_factor"]

        # Restore
        self.start_epoch = 1
        if args.checkpoint is not None:
            checkpoint_fullname = args.checkpoint
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            if 'model_state_dict' in checkpoint.keys():
                checkpoint_tmp = checkpoint['model_state_dict']
            else:
                checkpoint_tmp = checkpoint
            try:
                self.model.load_state_dict(checkpoint_tmp, strict=True)
            except:
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
            self.metric_logger = metric_logger(self.problem, self.model_params["dual_decoder"])

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
                        'metrics_logger': self.metric_logger
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

            print(" \n*** Test Done on {} *** ".format(self.args.problem))

            print(" \n*** Construction *** ")
            print(" NO-AUG SCORE: {}, Gap: {:.4f} ".format(self.val_metric_logger.construct_metrics["no_aug_score_list"],
                                                           self.val_metric_logger.construct_metrics["no_aug_gap_list"]))
            print(" AUGMENTATION SCORE: {}, Gap: {:.4f} ".format(self.val_metric_logger.construct_metrics["aug_score_list"],
                                                                 self.val_metric_logger.construct_metrics["aug_gap_list"]))
            print("Solution level Infeasible rate: {:.3f}%".format(self.val_metric_logger.construct_metrics["sol_infeasible_rate_list"]))
            print("Instance level Infeasible rate: {:.3f}%".format(self.val_metric_logger.construct_metrics["ins_infeasible_rate_list"]))

            print(" \n*** Inprovement *** ")
            print(" NO-AUG SCORE: {}, Gap: {:.4f} ".format(self.val_metric_logger.improve_metrics["no_aug_score_list"],
                                                           self.val_metric_logger.improve_metrics["no_aug_gap_list"]))
            print(" AUGMENTATION SCORE: {}, Gap: {:.4f} ".format(self.val_metric_logger.improve_metrics["aug_score_list"],
                                                                 self.val_metric_logger.improve_metrics["aug_gap_list"]))
            print("Solution level Infeasible rate: {:.3f}%".format(self.val_metric_logger.improve_metrics["sol_infeasible_rate_list"]))
            print("Instance level Infeasible rate: {:.3f}%".format(self.val_metric_logger.improve_metrics["ins_infeasible_rate_list"]))

            if not self.env_params["pomo_start"] and self.tester_params["test_z_sample_size"] == 0:
                self.env_params["pomo_size"] = init_pomo_size

            # log
            self._val_logger(epoch=1)

    def _train_one_epoch(self, epoch):
        episode = 0

        train_num_episode = self.trainer_params['train_episodes']
        total_step = math.floor(train_num_episode /self.trainer_params['train_batch_size'])
        batch_id = 0
        batch_reward = None
        weights = 0
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

                self._train_one_batch(data, env, batch_reward, weights, accumulation_step=accumulation_step)

                # torch.cuda.empty_cache()

                episode += batch_size
                batch_id += 1

                if episode >= train_num_episode:
                    break

        # Log Once, for each epoch
        self._print_log()

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
            except:
                self.model.pre_forward(reset_state, z)

            # Initialize the prob list
            if self.model_params["dual_decoder"]:
                prob_list1 = torch.zeros(size=(batch_size, env.pomo_size, 0)).to(self.device)
                prob_list2 = torch.zeros(size=(batch_size, env.pomo_size, 0)).to(self.device)
            else:
                prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0)).to(self.device)
                probs_return_list = torch.zeros(size=(batch_size, env.pomo_size, env.problem_size+1, 0)).to(self.device)
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

                state, reward, done, infeasible = env.step(selected.to(self.device),
                                                           out_reward=self.trainer_params["out_reward"],
                                                           soft_constrained=self.trainer_params["soft_constrained"],
                                                           backhaul_mask=self.trainer_params["backhaul_mask"],
                                                           penalty_normalize=self.trainer_params["penalty_normalize"])

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
        if "TSP" not in self.args.problem: self.metric_logger.dummy_size.update(env.dummy_size, batch_size)
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
            else:
                cons_reward = torch.stack(reward).sum(0)
            cons_log_prob = prob_list.log().sum(dim=2) if self.trainer_params["baseline"] == "share" else None # (batch, pomo)
            improve_loss, improve_reward, select_idx, best_solution, is_improved = self._improvement(env, cons_reward, batch_reward, weights, cons_log_prob)
            # if "TSP" not in self.args.problem: self.metric_logger.dummy_size.update(env.dummy_size, batch_size)
        else:
            improve_loss, improve_reward, select_idx, best_solution = 0.0, None, None, None
        # if self.rank==0: print(f"Rank {self.rank} >> improvement time: ", time.time() - tik)

        ###########################################Step & Return########################################
        if not self.model_params["improvement_only"]:
            construct_loss = self._get_construction_output(infeasible, reward, prob_list, improve_reward, select_idx, probs_return_list)
        else:
            construct_loss = 0.0, 0.0

        if self.trainer_params["imitation_learning"] and best_solution is not None:
            env.load_problems(batch_size, rollout_size=1, problems=data, aug_factor=1)
            reset_state, _, _ = env.reset()
            state, reward, done = env.pre_step()
            imit_prob_list = torch.zeros(size=(batch_size, 1, 0)).to(self.device)
            for step in range(best_solution.size(-1)):
                with torch.amp.autocast(device_type="cuda", enabled=amp_training):
                    _, prob, _ = self.model(state, pomo=self.env_params["pomo_start"], selected=best_solution[:,:,step],
                                                   candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None)
                imit_prob_list = torch.cat((imit_prob_list, prob[:, :, None]), dim=2)  # shape: (batch, pomo, solution)
                # shape: (batch, pomo)
                state, reward, done, infeasible = env.step(best_solution[:,:,step].to(self.device),
                                                           out_reward=self.trainer_params["out_reward"],
                                                           soft_constrained=self.trainer_params["soft_constrained"],
                                                           backhaul_mask=self.trainer_params["backhaul_mask"],
                                                           penalty_normalize=self.trainer_params["penalty_normalize"])
            imitation_loss = -(is_improved * imit_prob_list.mean(-1).mean(-1)).mean()
            self.metric_logger.construct_metrics["imitation_loss"].update(imitation_loss.item(), batch_size)
            self.metric_logger.construct_metrics["is_improved"].update(is_improved.sum()/batch_size, batch_size)

        if self.epoch == 1: self._print_log()
        if accumulation_step == 0:
            self.model.zero_grad()
            self.optimizer.zero_grad()
        ## way 1: static weights
        coefficient = 1. if self.trainer_params["improvement_only"] else 100.
        ## way 2: dynamic weights based on scale
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
            if self.trainer_params["reward_gating"] or self.trainer_params["adaptive_primal_dual"]:
                self.metric_logger.lambda_tw.update(self.lambda_[0].item())
                self.metric_logger.lambda_demand.update(self.lambda_[1].item())
                self.metric_logger.lambda_backhaul.update(self.lambda_[2].item())
                self.metric_logger.lambda_dl.update(self.lambda_[3].item())
        if self.trainer_params["imitation_learning"] and best_solution is not None:
            loss += imitation_loss * self.trainer_params["imitation_loss_weight"]
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

    def _val_one_batch(self, data, env, aug_factor=1, eval_type="argmax"):
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

        with torch.no_grad():
            if not (self.trainer_params["improvement_only"] and self.trainer_params["val_init_sol_strategy"] == "POMO"):
                env.load_problems(batch_size, rollout_size=rollout_size, problems=data, aug_factor=aug_factor)
                reset_state, _, _ = env.reset()

                state, reward, done = env.pre_step()
            ###########################################Construction########################################
            if self.rank==0: tik = time.time()
            if not self.model_params["improvement_only"]:
                if self.model_params["polynet"]:
                    z = self.sample_z_vectors(batch_size * aug_factor, starting_points, z_dim, z_sample_size,
                                              rollout_size)
                else:
                    z = None
                try:
                    self.model.module.pre_forward(reset_state, z)
                except:
                    self.model.pre_forward(reset_state, z)

                while not done:
                    selected, _, _ = self.model(state, pomo=self.env_params["pomo_start"],
                                                candidate_feature=env.node_tw_end if self.args.problem == "TSPTW" else None)
                    # shape: (batch, pomo)
                    state, reward, done, infeasible = env.step(selected,
                                                               soft_constrained = self.trainer_params["soft_constrained"],
                                                               backhaul_mask = self.trainer_params["backhaul_mask"])
                # Return
                self._get_construction_output_val(aug_factor, infeasible, reward)
            if self.rank==0: print(f"Rank {self.rank} >> val construction time: ", time.time() - tik)
            ###########################################Improvement########################################
            if self.trainer_params["validation_improve_steps"] > 0.:
                if self.model_params["clean_cache"]: torch.cuda.empty_cache()
                # self.model.decoder.k = None
                # self.model.decoder.v = None
                if self.rank==0: tik = time.time()
                if self.model_params["improvement_only"]:  # generate random solution
                    if self.trainer_params["val_init_sol_strategy"] != "POMO":
                        env.get_initial_solutions(strategy=self.trainer_params["init_sol_strategy"],
                                                   k=self.env_params["pomo_size"], max_dummy_size=self.trainer_params["max_dummy_size"])
                        self._get_construction_output_val(aug_factor, env.infeasible, env._get_travel_distance())
                    else:
                        self._get_pomo_initial_solution(env, data, batch_size, rollout_size, eval_type="argmax", aug_factor=aug_factor, val=True)
                self._val_improvement(env, aug_factor)
                if self.rank==0: print(f"Rank {self.rank} >> val improvement time: ", time.time() - tik)

    def _val_and_stat(self, dir, val_path, env, batch_size=500, val_episodes=1000, compute_gap=False, epoch=1):

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

            data = env.load_dataset(os.path.join(dir, val_path), offset=episode, num_samples=bs)

            self._val_one_batch(data, env, aug_factor=8, eval_type="argmax")

            episode += bs

            if self.tester_params["eval_only"]:
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, val_episodes)
                print("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}]".format(episode, val_episodes, elapsed_time_str, remain_time_str))

        self.val_metric_logger._log_output(self)

        try:
            sol_path = get_opt_sol_path(dir, env.problem, data[1].size(1))
        except:
            sol_path = os.path.join(dir, "lkh_" + val_path)

        compute_gap = os.path.exists(sol_path)

        if compute_gap:
            opt_sol = load_dataset(sol_path, disable_print=True)[: val_episodes]
            grid_factor = 100. if self.args.problem == "TSPTW" else 1.
            opt_sol = torch.tensor([i[0]/grid_factor for i in opt_sol])
            self.val_metric_logger._calculate_gap(self, opt_sol)
            try:
                if self.rank==0: print(f'Rank {self.rank} >> Val Score on {val_path}: [Construction] NO_AUG_Score: {self.val_metric_logger.construct_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.construct_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.construct_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.construct_metrics["aug_gap_list"]}%; Infeasible rate: {self.val_metric_logger.construct_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.construct_metrics["ins_infeasible_rate_list"]}% (instance-level)')
                if self.rank==0 and self.trainer_params["validation_improve_steps"] > 0.: print(f'Rank {self.rank} >> Val Score on {val_path}: [Improvement] NO_AUG_Score: {self.val_metric_logger.improve_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.improve_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.improve_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.improve_metrics["aug_gap_list"]}%; Infeasible rate: {self.val_metric_logger.improve_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.improve_metrics["ins_infeasible_rate_list"]}% (instance-level)')
            except:
                if self.rank==0: print(f'Rank {self.rank} >> Val Score on {val_path}: [Construction] NO_AUG_Score: {self.val_metric_logger.construct_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.construct_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.construct_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.construct_metrics["aug_gap_list"]}%')
                if self.rank == 0 and self.trainer_params["validation_improve_steps"] > 0.: print(f'Rank {self.rank} >> Val Score on {val_path}: [Improvement] NO_AUG_Score: {self.val_metric_logger.improve_metrics["no_aug_score_list"]}, NO_AUG_Gap: {self.val_metric_logger.improve_metrics["no_aug_gap_list"]}% --> AUG_Score: {self.val_metric_logger.improve_metrics["aug_score_list"]}, AUG_Gap: {self.val_metric_logger.improve_metrics["aug_gap_list"]}%')
        else:
            if self.rank==0: print(f'Rank {self.rank} >> Val Score on {val_path}: [Construction] NO_AUG_Score: {self.val_metric_logger.construct_metrics["no_aug_score_list"]}, --> AUG_Score: {self.val_metric_logger.construct_metrics["aug_score_list"]}; Infeasible rate: {self.val_metric_logger.construct_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.construct_metrics["ins_infeasible_rate_list"]}% (instance-level)')
            if self.rank==0 and self.trainer_params["validation_improve_steps"] > 0.: print(f'Rank {self.rank} >> Val Score on {val_path}: [Improvement] NO_AUG_Score: {self.val_metric_logger.improve_metrics["no_aug_score_list"]}, --> AUG_Score: {self.val_metric_logger.improve_metrics["aug_score_list"]}; Infeasible rate: {self.val_metric_logger.improve_metrics["sol_infeasible_rate_list"]}% (solution-level), {self.val_metric_logger.improve_metrics["ins_infeasible_rate_list"]}% (instance-level)')

    def _improvement(self, env, cons_reward=None, batch_reward=None, weights=None, cons_log_prob=None):
        amp_training = self.trainer_params['amp_training']
        # solution/rec shape: (batch, K, solution)
        solution = env.selected_node_list.clone() # shape: (batch, pomo, solution)
        if not self.trainer_params["improvement_only"]: # if yes, already generate k solutions for each instance
            # select top k
            solution, select_idx = select4improve(solution, cons_reward, strategy=self.trainer_params["select_strategy"],
                                                  K=self.trainer_params["select_top_k"], rnd_prob=self.trainer_params["stochastic_probability"],
                                                  diversity=self.trainer_params["diversity"])
            # solution shape: (batch, k, solution); solution_idx shape: (batch, k)
            feasibility_history = torch.gather(~env.infeasible, 1, select_idx)
        else:
            select_idx = torch.arange(env.pomo_size)[None, :].repeat(env.batch_size, 1)
            feasibility_history = ~env.infeasible

        if "TSP" not in self.args.problem: solution = get_solution_with_dummy_depot(solution, env.problem_size)
        batch_size, k, solution_size = solution.size()
        rec = sol2rec(solution).view(batch_size * k, -1)

        # preapare input
        obj, context, out_penalty, out_node_penalty = env.get_costs(rec, get_context=True, out_reward=self.trainer_params["out_reward"], penalty_factor = self.lambda_)
        obj = torch.cat((obj[:, None], obj[:, None], obj[:, None]), -1).clone()
        context2 = torch.zeros(batch_size * k, 9)
        context2[:, -1] = 1
        total_history = self.trainer_params["total_history"]
        feasibility_history = feasibility_history.view(-1, 1).expand(batch_size * k, total_history)
        action = None
        best_reward, best_index = (obj[:, 0].view(batch_size, k)).min(-1)
        rec_best = rec.view(batch_size, k, -1)[torch.arange(batch_size),best_index,:].clone()
        is_improved = torch.zeros(batch_size).bool()

        # sample trajectory
        t = 0
        T = self.trainer_params["improve_steps"]
        use_LoRA = False
        memory = Memory()
        while t < T:
            # print(">>>>>>>>>> ", t)
            entropy = []

            state = (env, rec, context, context2, action)
            if self.model_params["use_LoRA"] and t >= self.trainer_params["LoRA_begin_step"]: use_LoRA=True
            with torch.amp.autocast(device_type="cuda", enabled=amp_training):
                action, log_lh, entro_p, improvement_method  = self.model(state, solver="improvement", require_entropy=True, use_LoRA=use_LoRA)

            if self.model.training: memory.logprobs.append(log_lh.clone())
            entropy.append(entro_p)

            # state transient
            rec, rewards, obj, feasibility_history, context, context2, info, out_penalty, out_node_penalty = env.improvement_step(rec, action, obj, feasibility_history, t,
                                                                                             improvement_method = improvement_method,
                                                                                             weights=weights, out_reward = self.trainer_params["out_reward"],
                                                                                             penalty_factor=self.lambda_, penalty_normalize=self.trainer_params["penalty_normalize"], insert_before=self.trainer_params["insert_before"])

            # update best solution
            new_best, best_index = obj[:, 0].view(batch_size, k).min(-1)
            index = new_best < best_reward
            best_reward[index] = new_best[index] # update best reward
            is_improved = (is_improved | index)
            rec_best[index] = rec.view(batch_size, k, -1)[torch.arange(batch_size), best_index, :][index].clone() # update best solution

            if self.model.training: batch_reward.append(rewards[:, 0].clone())
            memory.rewards.append(rewards)
            memory.obj.append(obj.clone())
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
            soft_infeasible = (out_penalty.sum(0) <= env.epsilon) & (out_penalty.sum(0) > 0.)
            memory.feasible.append(feasible)
            memory.soft_feasible.append(soft_infeasible)

            # next
            t = t + 1

        # calculate improvement loss
        if self.model.training:
            if self.trainer_params["baseline"] != "share":
                log_prob = torch.stack(memory.logprobs).view(T, batch_size, k)
                reward = torch.stack(memory.rewards).sum(-1).view(T, batch_size, k)
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
                # else:
                loss_mean = loss.mean()
                self.metric_logger.improve_metrics["loss"].update(loss_mean.item(), batch_size)
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
        if self.trainer_params["bonus_for_construction"] and self.trainer_params["baseline"] != "improve":
            improve_reward = score[:, :, 0].min(dim=0)[0].view(batch_size, k) # output the bsf during improvemnet for every initial solutions
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
            min_out_node_penalty = out_node_penalty[0].min(dim=1)[0].float().mean()  # get best results from pomo
            self.metric_logger.improve_metrics["tw_out"].update(min_out_penalty.item(), batch_size)
            self.metric_logger.improve_metrics["tw_out_nodes"].update(min_out_node_penalty.item(), batch_size)
            min_out_penalty = out_penalty[:,1].min(dim=1)[0].float().mean()  # get best results from pomo
            min_out_node_penalty = out_node_penalty[1].min(dim=1)[0].float().mean()  # get best results from pomo
            self.metric_logger.improve_metrics["capacity_out"].update(min_out_penalty.item(), batch_size)
            self.metric_logger.improve_metrics["capacity_out_nodes"].update(min_out_node_penalty.item(), batch_size)
            min_out_penalty = out_penalty[:,2].min(dim=1)[0].float().mean()  # get best results from pomo
            min_out_node_penalty = out_node_penalty[2].min(dim=1)[0].float().mean()  # get best results from pomo
            self.metric_logger.improve_metrics["backhaul_out"].update(min_out_penalty.item(), batch_size)
            self.metric_logger.improve_metrics["backhaul_out_nodes"].update(min_out_node_penalty.item(), batch_size)
            min_out_penalty = out_penalty[:,3].min(dim=1)[0].float().mean()  # get best results from pomo
            min_out_node_penalty = out_node_penalty[3].min(dim=1)[0].float().mean()  # get best results from pomo
            self.metric_logger.improve_metrics["dlout"].update(min_out_penalty.item(), batch_size)
            self.metric_logger.improve_metrics["dlout_nodes"].update(min_out_node_penalty.item(), batch_size)
            out_penalty = out_penalty.sum(1)
            out_node_penalty = out_node_penalty.sum(1)
        min_out_penalty = out_penalty.min(dim=1)[0].float().mean()  # get best results from pomo
        min_out_node_penalty = out_node_penalty.min(dim=1)[0].float().mean()  # get best results from pomo
        self.metric_logger.improve_metrics["out"].update(min_out_penalty.item(), batch_size)
        self.metric_logger.improve_metrics["out_nodes"].update(min_out_node_penalty.item(), batch_size)

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

        # end update
        memory.clear_memory()

        return loss_mean, improve_reward, select_idx, remove_dummy_depot_from_solution(rec2sol(rec_best).view(batch_size, 1, -1), env.problem_size), is_improved

    def _val_improvement(self, env, aug_factor):

        with torch.no_grad():
            # solution/rec shape: (batch, pomo, solution)
            solution = env.selected_node_list.clone()
            if "TSP" not in self.args.problem: solution = get_solution_with_dummy_depot(solution, env.problem_size)
            batch_size, pomo_size, solution_size = solution.size() # batch_size = aug_factor * batch_size
            rec = sol2rec(solution).view(batch_size * pomo_size, -1)

            # preapare input
            obj, context, out_penalty, out_node_penalty = env.get_costs(rec, get_context=True) # obj only
            obj = torch.cat((obj[:, None], obj[:, None], obj[:, None]), -1).clone()
            # obj = obj.unsqueeze(-1).expand(-1, -1, 3)
            context2 = torch.zeros(batch_size * pomo_size, 9)
            context2[:, -1] = 1
            total_history = self.trainer_params["total_history"]
            feasibility_history = (~env.infeasible).view(-1, 1).expand(batch_size * pomo_size, total_history)
            action = None

            # sample trajectory
            t = 0
            T = self.trainer_params["validation_improve_steps"]
            use_LoRA = False
            # memory = Memory()
            # initial solution from construction
            feasible_all = ((~env.infeasible).view(-1)).int()
            min_scores = torch.full((batch_size * pomo_size,), float('inf'))
            min_scores = torch.where(feasible_all.bool(), obj[:, 0], min_scores)
            out_node_penalties = torch.full((batch_size * pomo_size,), float('inf'))
            # if self.args.problem == "CVRP":
            #     out_node_penalty = (context[2] > 1.00001).sum(-1)
            # elif self.args.problem == "TSPTW":
            #     out_node_penalty = (torch.clamp_min(context[1] - context[-1], 0.0) > 1e-5).sum(-1) # (b*k)
            out_node_penalties = torch.where(~(feasible_all.bool()), out_node_penalty, out_node_penalties)
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
            out_penalties = torch.where(~(feasible_all.bool()), out_penalty, out_penalties)
            del out_penalty

            while t < T:
                # print(t)

                state = (env, rec, context, context2, action)

                if self.model_params["use_LoRA"] and t >= self.trainer_params["LoRA_begin_step"]: use_LoRA = True
                action, _, improvement_method = self.model(state, solver="improvement", require_entropy=False, use_LoRA=use_LoRA)

                # state transient
                # rec, rewards, obj, feasibility_history, context, context2, info
                rec, _, obj, feasibility_history, context, context2, _, out_penalty, out_node_penalty = env.improvement_step(rec, action, obj, feasibility_history, t,
                                                                                                                       improvement_method = improvement_method, insert_before=self.trainer_params["insert_before"])

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
                min_scores = torch.where(feasible, torch.min(min_scores, obj[:, 0]), min_scores)

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
                out_node_penalties = torch.min(out_node_penalties, out_node_penalty.sum(0))
                out_penalties = torch.min(out_penalties, out_penalty.sum(0))

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

    def _get_construction_output(self, infeasible, reward, prob_list=None, improve_reward=None, select_idx=None, probs_return_list=None):
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
                    if self.trainer_params["baseline"] == "group":
                        baseline = reward.float().mean(dim=1, keepdims=True)
                        advantage = reward - baseline  # (batch, pomo)
                        if self.trainer_params["bonus_for_construction"] and improve_reward is not None:
                            after_improve_reward = -reward.clone()
                            after_improve_reward.scatter_(dim=1, index=select_idx, src=improve_reward)
                            new_advantage = advantage / 10
                            new_advantage = torch.where(after_improve_reward < -reward, advantage, new_advantage)
                            advantage = new_advantage.clone()
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
                if self.trainer_params["diversity_loss"]:
                    self.metric_logger.construct_metrics["construct_RL_loss"].update(loss.mean().item(), batch_size)
                    if probs_return_list is None:
                        # implementation 1: only focus on the entropy on the probs of the selected nodes
                        diversity_loss = -(prob_list * prob_list.log()).sum(dim=2)  # Entropy
                    else:
                        # implementation 2: increase diversity for the whole action probability distributions
                        diversity_loss = -(probs_return_list * probs_return_list.log()).sum(dim=2).mean(dim=-1) # b * p
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

    def _get_construction_output_val(self, aug_factor, infeasible, reward):
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

        if self.trainer_params["fsb_dist_only"]:
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
            fsb_no_aug = reward_masked[0,:,:].max(dim=1, keepdim=True).values
            fsb_aug = reward_masked.max(dim=0).values.max(dim=-1).values
            no_aug_score, aug_score = -fsb_no_aug, -fsb_aug
        else:
            max_pomo_reward, _ = aug_reward.max(dim=2)
            no_aug_score = -max_pomo_reward[0, :].float()
            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
            aug_score = -max_aug_pomo_reward.float()
            infeasible_output = infeasible

        self.val_metric_logger._construct_tensor_update("no_aug_score", no_aug_score)
        self.val_metric_logger._construct_tensor_update("aug_score", aug_score)

    def _print_log(self):
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
    def _log_in_tb_logger(self, epoch):

        self.tb_logger.log_value('construction/train_score', self.metric_logger.construct_metrics["score"].avg, epoch)
        self.tb_logger.log_value('construction/train_loss', self.metric_logger.construct_metrics["loss"].avg, epoch)
        self.tb_logger.log_value('construction/construct_RL_loss', self.metric_logger.construct_metrics["construct_RL_loss"].avg, epoch)
        self.tb_logger.log_value('construction/diversity_loss', self.metric_logger.construct_metrics["diversity_loss"].avg, epoch)
        self.tb_logger.log_value('construction/max_vehicle_number', self.metric_logger.dummy_size.avg, epoch)
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
            if self.trainer_params["fsb_dist_only"]:
                self.tb_logger.log_value("improvement_feasibility/feasible_dist_mean", self.metric_logger.improve_metrics["feasible_dist_mean"].avg, epoch)
                self.tb_logger.log_value("improvement_feasibility/feasible_dist_max_pomo_mean", self.metric_logger.improve_metrics["feasible_dist_max_pomo_mean"].avg, epoch)
                self.tb_logger.log_value("improvement_feasibility/epsilon_feasible_dist_mean", self.metric_logger.improve_metrics["epsilon_feasible_dist_mean"].avg, epoch)
                self.tb_logger.log_value("improvement_feasibility/epsilon_feasible_dist_max_pomo_mean", self.metric_logger.improve_metrics["epsilon_feasible_dist_max_pomo_mean"].avg, epoch)
            # if self.model_params["dual_decoder"]:
            #     self.tb_logger.log_value('construction/train_loss_dist', self.metric_logger.improve_metrics["loss1"].avg, epoch)
            #     self.tb_logger.log_value('construction/train_loss_timeout', self.metric_logger.improve_metrics["loss2"].avg, epoch)

    def _log_in_wandb(self, epoch):
        log_data = {
            'construction/train_score': self.metric_logger.construct_metrics["score"].avg,
            'construction/train_loss': self.metric_logger.construct_metrics["loss"].avg,
            'construction/construct_RL_loss': self.metric_logger.construct_metrics["construct_RL_loss"].avg,
            'construction/diversity_loss': self.metric_logger.construct_metrics["diversity_loss"].avg,
            'construction/max_vehicle_number': self.metric_logger.dummy_size.avg,
            'coefficient': self.metric_logger.coefficient.avg,
            'sigma1': self.metric_logger.sigma1.avg,
            'sigma2': self.metric_logger.sigma2.avg,
            'lambda_tw': self.metric_logger.lambda_tw.avg,
            'lambda_demand': self.metric_logger.lambda_demand.avg,
            'lambda_backhaul': self.metric_logger.lambda_backhaul.avg,
            'lambda_dl': self.metric_logger.lambda_dl.avg,
            'construction_feasibility/solution_infeasible_rate': self.metric_logger.construct_metrics[
                "sol_infeasible_rate"].avg,
            'construction_feasibility/instance_infeasible_rate': self.metric_logger.construct_metrics[
                "ins_infeasible_rate"].avg
        }

        if self.trainer_params["out_reward"]:
            log_data.update({
                'construction_feasibility/total_out': self.metric_logger.construct_metrics["out"].avg,
                'construction_feasibility/out_nodes': self.metric_logger.construct_metrics["out_nodes"].avg,
            })
            if self.args.problem in ["OVRPBLTW", "OVRPLTW", "VRPBLTW"]:
                log_data.update({
                    'construction_feasibility/dlout': self.metric_logger.construct_metrics["dlout"].avg,
                    'construction_feasibility/dlout_nodes': self.metric_logger.construct_metrics[
                        "dlout_nodes"].avg,
                    'construction_feasibility/capacity_out': self.metric_logger.construct_metrics[
                        "capacity_out"].avg,
                    'construction_feasibility/capacity_out_nodes': self.metric_logger.construct_metrics[
                        "capacity_out_nodes"].avg,
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
                'improvement/train_score_epsilon_fsb_bsf': self.metric_logger.improve_metrics[
                    "epsilon_fsb_bsf_score"].avg,
                'improvement/train_improve_reward': self.metric_logger.improve_metrics["improve_reward"].avg,
                'improvement/train_reg_reward': self.metric_logger.improve_metrics["reg_reward"].avg,
                'improvement/train_bonus_reward': self.metric_logger.improve_metrics[
                    "bonus_reward"].avg,
                'improvement/train_loss': self.metric_logger.improve_metrics["loss"].avg,
                'improvement/entropy': self.metric_logger.improve_metrics["entropy"].avg,
                'improvement_feasibility/solution_infeasible_rate': self.metric_logger.improve_metrics[
                    "sol_infeasible_rate"].avg,
                'improvement_feasibility/instance_infeasible_rate': self.metric_logger.improve_metrics[
                    "ins_infeasible_rate"].avg,
                'improvement_feasibility/epsilon_solution_infeasible_rate': self.metric_logger.improve_metrics[
                    "soft_sol_infeasible_rate"].avg,
                'improvement_feasibility/epsilon_instance_infeasible_rate': self.metric_logger.improve_metrics[
                    "soft_ins_infeasible_rate"].avg,
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
                    })

            if self.trainer_params["fsb_dist_only"]:
                log_data.update({
                    'improvement_feasibility/feasible_dist_mean': self.metric_logger.improve_metrics[
                        "feasible_dist_mean"].avg,
                    'improvement_feasibility/feasible_dist_max_pomo_mean': self.metric_logger.improve_metrics[
                        "feasible_dist_max_pomo_mean"].avg,
                    'improvement_feasibility/epsilon_feasible_dist_mean': self.metric_logger.improve_metrics[
                        "epsilon_feasible_dist_mean"].avg,
                    'improvement_feasibility/epsilon_feasible_dist_max_pomo_mean':
                        self.metric_logger.improve_metrics["epsilon_feasible_dist_max_pomo_mean"].avg,
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
        if not self.trainer_params["improvement_only"]:
            score = self.val_metric_logger.construct_metrics["aug_score_list"]
            gap = self.val_metric_logger.construct_metrics["aug_gap_list"]
            sol_infeasible_rate_list = self.val_metric_logger.construct_metrics["sol_infeasible_rate_list"]
            ins_infeasible_rate_list = self.val_metric_logger.construct_metrics["ins_infeasible_rate_list"]

        if self.trainer_params["validation_improve_steps"] > 0.:
            improve_score = self.val_metric_logger.improve_metrics["aug_score_list"]
            improve_gap = self.val_metric_logger.improve_metrics["aug_gap_list"]
            improve_sol_infeasible_rate_list = self.val_metric_logger.improve_metrics["sol_infeasible_rate_list"]
            improve_ins_infeasible_rate_list = self.val_metric_logger.improve_metrics["ins_infeasible_rate_list"]

        if self.args.multiple_gpu:
            dist.barrier()
            score = gather_tensor_and_concat(torch.tensor([score]).cuda())
            gap = gather_tensor_and_concat(torch.tensor([gap]).cuda())
            sol_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([sol_infeasible_rate_list]).cuda())
            ins_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([ins_infeasible_rate_list]).cuda())
            if self.trainer_params["validation_improve_steps"] > 0.:
                improve_score = gather_tensor_and_concat(torch.tensor([improve_score]).cuda())
                improve_gap = gather_tensor_and_concat(torch.tensor([improve_gap]).cuda())
                improve_sol_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([improve_sol_infeasible_rate_list]).cuda())
                improve_ins_infeasible_rate_list = gather_tensor_and_concat(torch.tensor([improve_ins_infeasible_rate_list]).cuda())
            dist.barrier()
            score = score.mean()
            gap = gap.mean()
            ins_infeasible_rate_list = ins_infeasible_rate_list.mean()
            sol_infeasible_rate_list = sol_infeasible_rate_list.mean()
            if self.trainer_params["validation_improve_steps"] > 0.:
                improve_score = improve_score.mean()
                improve_gap = improve_gap.mean()
                improve_sol_infeasible_rate_list = improve_sol_infeasible_rate_list.mean()
                improve_ins_infeasible_rate_list = improve_ins_infeasible_rate_list.mean()

        if self.rank == 0:
            if not self.trainer_params["improvement_only"]:
                self.result_log["val_score"].append(score)
                self.result_log["val_gap"].append(gap)
                self.result_log["val_infsb_rate"].append([sol_infeasible_rate_list, ins_infeasible_rate_list])
            if self.trainer_params["validation_improve_steps"] > 0.:
                self.result_log["val_score_improve"].append(improve_score)
                self.result_log["val_gap_improve"].append(improve_gap)
                self.result_log["val_infsb_rate_improve"].append([improve_sol_infeasible_rate_list, improve_ins_infeasible_rate_list])

        if self.args.wandb_logger and self.rank == 0:
            logdata = {}
            if not self.trainer_params["improvement_only"]:
                logdata ={"val/val_score": score,
                            "val/val_gap": gap,
                            'val/val_sol_infsb_rate': sol_infeasible_rate_list,
                            'val/val_ins_infsb_rate': ins_infeasible_rate_list}
            if self.trainer_params["validation_improve_steps"] > 0.:
                logdata.update({"val/improve_val_score": improve_score,
                                "val/improve_val_gap": improve_gap,
                                'val/improve_val_sol_infsb_rate': improve_sol_infeasible_rate_list,
                                'val/improve_val_ins_infsb_rate': improve_ins_infeasible_rate_list})
            wandb.log(logdata, step = epoch)

        if self.tb_logger and self.rank == 0:
            if not self.trainer_params["improvement_only"]:
                self.tb_logger.log_value('val/val_score', score, epoch)
                self.tb_logger.log_value('val/val_gap', gap, epoch)
            if self.trainer_params["validation_improve_steps"] > 0.:
                self.tb_logger.log_value('val/improve_val_score', improve_score, epoch)
                self.tb_logger.log_value('val/improve_val_gap', improve_gap, epoch)
            try:
                if not self.trainer_params["improvement_only"]:
                    self.tb_logger.log_value('val/val_sol_infsb_rate', sol_infeasible_rate_list, epoch)
                    self.tb_logger.log_value('val/val_ins_infsb_rate', ins_infeasible_rate_list, epoch)
                if self.trainer_params["validation_improve_steps"] > 0.:
                    self.tb_logger.log_value('val/improve_val_sol_infsb_rate', improve_sol_infeasible_rate_list, epoch)
                    self.tb_logger.log_value('val/improve_val_ins_infsb_rate', improve_ins_infeasible_rate_list, epoch)
            except:
                pass

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
                state, reward, done, infeasible = env.step(selected,
                                                           out_reward=self.trainer_params["out_reward"] if not val else False,
                                                           soft_constrained=self.trainer_params["soft_constrained"],
                                                           backhaul_mask=self.trainer_params["backhaul_mask"])

        if not val:
            self._get_construction_output(infeasible, reward)
            return torch.stack(reward).sum(0)
        else:
            self._get_construction_output_val(aug_factor, infeasible, reward)

