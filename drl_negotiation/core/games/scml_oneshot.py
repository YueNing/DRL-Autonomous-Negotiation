import time
import copy
import sys
import random
import numpy as np
import traceback
from pprint import pprint
from typing import Optional, List, Tuple
from collections import defaultdict
from negmas.situated import Operations, Entity, Agent, Contract
from negmas.helpers import exception2str
from negmas.events import Event
from negmas.mechanisms import Mechanism
from drl_negotiation.core.games._game import TrainingWorld
from scml.oneshot import SCML2020OneShotWorld
from scml.scml2020 import is_system_agent

__all__ = [
    "TrainWorld"
]


class TrainWorld(TrainingWorld):
    def __init__(self, world: SCML2020OneShotWorld):
        super(TrainWorld, self).__init__(world)

        # rollout
        self.tmp_obs = None
        self.tmp_state = None
        self.tmp_actions = None
        self.tmp_actions_onehot = None
        self.tmp_avail_actions = None
        self.tmp_terminated = False

        self.set_agent = []

    @property
    def n_negotiations(self):
        return self.world._World__n_negotiations

    @n_negotiations.setter
    def n_negotiations(self, negotiations):
        self.world._World__n_negotiations = negotiations

    @property
    def n_contracts_dropped(self):
        """return private _World__n_contracts_dropped"""
        return self.world._World__n_contracts_dropped

    @n_contracts_dropped.setter
    def n_contracts_dropped(self, dropped):
        self.world._World__n_contracts_dropped = dropped

    @property
    def n_contracts_cancelled(self):
        return self.world._World__n_contracts_cancelled

    @n_contracts_cancelled.setter
    def n_contracts_cancelled(self, cancelled):
        self.world._World__n_contracts_cancelled = cancelled

    @property
    def n_contracts_concluded(self):
        return self.world._World__n_contracts_concluded

    @n_contracts_concluded.setter
    def n_contracts_concluded(self, concluded):
        self.world._World__n_contracts_concluded = concluded

    @property
    def n_contracts_signed(self):
        return self.world._World__n_contracts_signed

    @n_contracts_signed.setter
    def n_contracts_signed(self, singed):
        self.world._World__n_contracts_signed = singed

    @property
    def policy_agents(self):
        agents = {}
        for agent_id, agent in self.world.agents.items():
            if is_system_agent(agent_id):
                pass
            elif agent_id[2:].startswith("My"):
                agents[agent_id] = agent
        return agents

    def reset(self, world):
        """reset the world"""
        self.world = world

    def t_step(self):
        self.world.step()

    def step(self) -> bool:
        """A single simulation step"""
        # self.world.step()
        if self.world._start_time is None or self.world._start_time < 0:
            self.world._start_time = time.perf_counter()
        if self.world.time >= self.world.time_limit:
            self.tmp_terminated = True
            return False
        self.world._n_negs_per_agent_per_step = defaultdict(int)
        if self.world.current_step >= self.world.n_steps:
            return False
        did_not_start, self.world._started = self.world._started, True
        if self.world.current_step == 0:
            self.world._sim_start = time.perf_counter()
            self.world._step_start = self.world._sim_start
            for priority in sorted(self.world._entities.keys()):
                for agent in self.world._entities[priority]:
                    self.world.call(agent, agent.init_)
                    if self.world.time >= self.world.time_limit:
                        self.tmp_terminated = True
                        return False
            # update monitors
            for monitor in self.world.stats_monitors:
                if self.world.safe_stats_monitoring:
                    __stats = copy.deepcopy(self.world.stats)
                else:
                    __stats = self.world.stats
                monitor.init(__stats, world_name=self.world.name)
            for monitor in self.world.world_monitors:
                monitor.init(self.world)
        else:
            self.world._step_start = time.perf_counter()
        # do checkpoint processing
        self.world.checkpoint_on_step_started()

        for agent in self.world.agents.values():
            self.world.call(agent, agent.on_simulation_step_started)
            if self.world.time >= self.world.time_limit:
                self.tmp_terminated = True
                return False

        self.world.loginfo(
            f"{len(self.world._negotiations)} Negotiations/{len(self.world.agents)} Agents"
        )

        # initialize stats
        # ----------------
        n_new_contract_executions = 0
        n_new_breaches = 0
        n_new_contract_errors = 0
        n_new_contract_nullifications = 0
        activity_level = 0
        n_steps_broken, n_steps_success = 0, 0
        n_broken, n_success = 0, 0
        stage = 0
        stats_stage = 0
        blevel = 0.0

        _n_registered_negotiations_before = len(self.world._negotiations)

        def _run_negotiations(n_steps: Optional[int] = None):
            """ Runs all bending negotiations """
            nonlocal n_steps_broken, n_steps_success, n_broken, n_success
            mechanisms = list(
                (_.mechanism, _.partners)
                for _ in self.world._negotiations.values()
                if _ is not None
            )
            (
                _,
                _,
                n_steps_broken_,
                n_steps_success_,
                n_broken_,
                n_success_,
            ) = self._step_negotiations(
                [_[0] for _ in mechanisms], n_steps, False, [_[1] for _ in mechanisms]
            )
            if self.world.time >= self.world.time_limit:
                self.tmp_terminated = True
                return
            n_total_broken = n_broken + n_broken_
            if n_total_broken > 0:
                n_steps_broken = (
                                         n_steps_broken * n_broken + n_steps_broken_ * n_broken_
                                 ) / n_total_broken
                n_broken = n_total_broken
            n_total_success = n_success + n_success_
            if n_total_success > 0:
                n_steps_success = (
                                          n_steps_success * n_success + n_steps_success_ * n_success_
                                  ) / n_total_success
                n_success = n_total_success

        def _step_agents():
            # Step all entities in the world once:
            # ------------------------------------
            # note that entities are simulated in the partial-order specified by their priority value
            tasks: List[Entity] = []
            for priority in sorted(self.world._entities.keys()):
                tasks += [_ for _ in self.world._entities[priority]]

            for task in tasks:
                self.world.call(task, task.step_)
                if self.world.time >= self.world.time_limit:
                    self.tmp_terminated = True
                    break

        def _sign_contracts():
            self.world._process_unsigned()

        def _simulation_step():
            nonlocal stage
            try:
                self.world.simulation_step(stage)
                if self.world.time >= self.world.time_limit:
                    self.tmp_terminated = True
                    return
            except Exception as e:
                self.world.simulation_exceptions[self.world._current_step].append(exception2str())
                if not self.world.ignore_simulation_exceptions:
                    raise (e)
            stage += 1

        def _execute_contracts():
            # execute contracts that are executable at this step
            # --------------------------------------------------
            nonlocal n_new_breaches, n_new_contract_executions, n_new_contract_errors, n_new_contract_nullifications, activity_level, blevel
            current_contracts = [
                _ for _ in self.world.executable_contracts() if _.nullified_at < 0
            ]
            if len(current_contracts) > 0:
                # remove expired contracts
                executed = set()
                current_contracts = self.world.order_contracts_for_execution(
                    current_contracts
                )

                for contract in current_contracts:
                    if self.world.time >= self.world.time_limit:
                        self.tmp_terminated = True
                        break
                    if contract.signed_at < 0:
                        continue
                    try:
                        contract_breaches = self.world.start_contract_execution(contract)
                    except Exception as e:
                        for p in contract.partners:
                            self.world.contracts_erred[p] += 1
                        self.world.contract_exceptions[self.world._current_step].append(
                            exception2str()
                        )
                        contract.executed_at = self.world.current_step
                        self.world._saved_contracts[contract.id]["breaches"] = ""
                        self.world._saved_contracts[contract.id]["executed_at"] = -1
                        self.world._saved_contracts[contract.id]["dropped_at"] = -1
                        self.world._saved_contracts[contract.id]["nullified_at"] = -1
                        self.world._saved_contracts[contract.id][
                            "erred_at"
                        ] = self.world._current_step
                        self.world._add_edges(
                            contract.partners[0],
                            contract.partners,
                            self.world._edges_contracts_erred,
                            bi=True,
                        )
                        n_new_contract_errors += 1
                        if not self.world.ignore_contract_execution_exceptions:
                            raise e
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        self.world.logerror(
                            f"Contract exception @{str(contract)}: "
                            f"{traceback.format_tb(exc_traceback)}",
                            Event("contract-exception", dict(contract=contract, exception=e)),
                        )
                        continue
                    if contract_breaches is None:
                        for p in contract.partners:
                            self.world.contracts_nullified[p] += 1
                        self.world._saved_contracts[contract.id]["breaches"] = ""
                        self.world._saved_contracts[contract.id]["executed_at"] = -1
                        self.world._saved_contracts[contract.id]["dropped_at"] = -1
                        self.world._saved_contracts[contract.id][
                            "nullified_at"
                        ] = self.world._current_step
                        self.world._add_edges(
                            contract.partners[0],
                            contract.partners,
                            self.world._edges_contracts_nullified,
                            bi=True,
                        )
                        self.world._saved_contracts[contract.id]["erred_at"] = -1
                        n_new_contract_nullifications += 1
                        self.world.loginfo(
                            f"Contract nullified: {str(contract)}",
                            Event("contract-nullified", dict(contract=contract)),
                        )
                    elif len(contract_breaches) < 1:
                        for p in contract.partners:
                            self.world.contracts_executed[p] += 1
                        self.world._saved_contracts[contract.id]["breaches"] = ""
                        self.world._saved_contracts[contract.id]["dropped_at"] = -1
                        self.world._saved_contracts[contract.id][
                            "executed_at"
                        ] = self.world._current_step
                        self.world._add_edges(
                            contract.partners[0],
                            contract.partners,
                            self.world._edges_contracts_executed,
                            bi=True,
                        )
                        self.world._saved_contracts[contract.id]["nullified_at"] = -1
                        self.world._saved_contracts[contract.id]["erred_at"] = -1
                        executed.add(contract)
                        n_new_contract_executions += 1
                        _size = self.world.contract_size(contract)
                        if _size is not None:
                            activity_level += _size
                        for partner in contract.partners:
                            self.world.call(
                                self.world.agents[partner],
                                self.world.agents[partner].on_contract_executed,
                                contract,
                            )
                            if self.world.time >= self.world.time_limit:
                                self.tmp_terminated = True
                                break
                    else:
                        for p in contract.partners:
                            self.world.contracts_breached[p] += 1
                        self.world._saved_contracts[contract.id]["executed_at"] = -1
                        self.world._saved_contracts[contract.id]["nullified_at"] = -1
                        self.world._saved_contracts[contract.id]["dropped_at"] = -1
                        self.world._saved_contracts[contract.id]["erred_at"] = -1
                        self.world._saved_contracts[contract.id]["breaches"] = "; ".join(
                            f"{_.perpetrator}:{_.type}({_.level})"
                            for _ in contract_breaches
                        )
                        breachers = set(
                            (_.perpetrator, tuple(_.victims)) for _ in contract_breaches
                        )
                        for breacher, victims in breachers:
                            if isinstance(victims, str) or isinstance(victims, Agent):
                                victims = [victims]
                            self.world._add_edges(
                                breacher,
                                victims,
                                self.world._edges_contracts_breached,
                                bi=False,
                            )
                        for b in contract_breaches:
                            self.world._saved_breaches[b.id] = b.as_dict()
                            self.world.loginfo(
                                f"Breach of {str(contract)}: {str(b)} ",
                                Event("contract-breached", dict(contract=contract, breach=b)),
                            )
                        resolution = self.world._process_breach(
                            contract, list(contract_breaches)
                        )
                        if resolution is None:
                            n_new_breaches += 1
                            blevel += sum(_.level for _ in contract_breaches)
                        else:
                            n_new_contract_executions += 1
                            self.world.loginfo(
                                f"Breach resolution cor {str(contract)}: {str(resolution)} ",
                                Event("breach-resolved", dict(contract=contract, breaches=list(contract_breaches),
                                                              resolution=resolution)),
                            )
                        self.world.complete_contract_execution(
                            contract, list(contract_breaches), resolution
                        )
                        self.world.loginfo(
                            f"Executed {str(contract)}",
                            Event("contract-executed", dict(contract=contract)),
                        )
                        for partner in contract.partners:
                            self.world.call(
                                self.world.agents[partner],
                                self.world.agents[partner].on_contract_breached,
                                contract,
                                list(contract_breaches),
                                resolution,
                            )
                            if self.world.time >= self.world.time_limit:
                                self.tmp_terminated = True
                                break
                    contract.executed_at = self.world.current_step
            dropped = self.world.get_dropped_contracts()
            self.world.delete_executed_contracts()  # note that all contracts even breached ones are to be deleted
            for c in dropped:
                self.world.loginfo(
                    f"Dropped {str(c)}",
                    Event("dropped-contract", dict(contract=c)),
                )
                self.world._saved_contracts[c.id]["dropped_at"] = self.world._current_step
                for p in c.partners:
                    self.world.contracts_dropped[p] += 1
            self.n_contracts_dropped += len(dropped)

        def _stats_update():
            nonlocal stats_stage
            self.world.update_stats(stats_stage)
            stats_stage += 1

        operation_map = {
            Operations.AgentSteps       : _step_agents,
            Operations.ContractExecution: _execute_contracts,
            Operations.ContractSigning  : _sign_contracts,
            Operations.Negotiations     : _run_negotiations,
            Operations.SimulationStep   : _simulation_step,
            Operations.StatsUpdate      : _stats_update,
        }

        for operation in self.world.operations:
            operation_map[operation]()
            if self.world.time >= self.world.time_limit:
                self.tmp_terminated = True
                return False

        # remove all negotiations that are completed
        # ------------------------------------------
        completed = list(
            k
            for k, _ in self.world._negotiations.items()
            if _ is not None and _.mechanism.completed
        )
        for key in completed:
            self.world._negotiations.pop(key, None)

        # update stats
        # ------------
        self.world._stats["n_registered_negotiations_before"].append(
            _n_registered_negotiations_before
        )
        self.world._stats["n_contracts_executed"].append(n_new_contract_executions)
        self.world._stats["n_contracts_erred"].append(n_new_contract_errors)
        self.world._stats["n_contracts_nullified"].append(n_new_contract_nullifications)
        self.world._stats["n_contracts_cancelled"].append(self.n_contracts_cancelled)
        self.world._stats["n_contracts_dropped"].append(self.n_contracts_dropped)
        self.world._stats["n_breaches"].append(n_new_breaches)
        self.world._stats["breach_level"].append(blevel)
        self.world._stats["n_contracts_signed"].append(self.n_contracts_signed)
        self.world._stats["n_contracts_concluded"].append(self.n_contracts_concluded)
        self.world._stats["n_negotiations"].append(self.n_negotiations)
        self.world._stats["n_negotiation_rounds_successful"].append(n_steps_success)
        self.world._stats["n_negotiation_rounds_failed"].append(n_steps_broken)
        self.world._stats["n_negotiation_successful"].append(n_success)
        self.world._stats["n_negotiation_failed"].append(n_broken)
        self.world._stats["n_registered_negotiations_after"].append(len(self.world._negotiations))
        self.world._stats["activity_level"].append(activity_level)
        current_time = time.perf_counter() - self.world._step_start
        self.world._stats["step_time"].append(current_time)
        total = self.world._stats.get("total_time", [0.0])[-1]
        self.world._stats["total_time"].append(total + current_time)
        self.n_negotiations = 0
        self.n_contracts_signed = 0
        self.n_contracts_concluded = 0
        self.n_contracts_cancelled = 0
        self.n_contracts_dropped = 0

        self.world.append_stats()
        for agent in self.world.agents.values():
            self.world.call(agent, agent.on_simulation_step_ended)
            if self.world.time >= self.world.time_limit:
                self.tmp_terminated = True
                return False

        for monitor in self.world.stats_monitors:
            if self.world.safe_stats_monitoring:
                __stats = copy.deepcopy(self.world.stats)
            else:
                __stats = self.world.stats
            monitor.step(__stats, world_name=self.world.name)
        for monitor in self.world.world_monitors:
            monitor.step(self.world)

        self.world._current_step += 1
        self.world.frozen_time = self.world.time
        # always indicate that the simulation is to continue
        return True

    def _step_negotiations(
            self,
            mechanisms: List[Mechanism],
            n_steps: Optional[int],
            force_immediate_signing,
            partners: List[Agent],
    ) -> Tuple[List[None], list, int, int, int, int]:
        """TODO coding here, observation, stat, reward, action"""
        # return self.world._step_negotiations(
        #     mechanisms,
        #     n_steps,
        #     force_immediate_signing,
        #     partners
        # )
        """ Runs all bending negotiations """
        running = [_ is not None for _ in mechanisms]
        contracts = [None] * len(mechanisms)
        indices = list(range(len(mechanisms)))
        n_steps_broken_, n_steps_success_ = 0, 0
        n_broken_, n_success_ = 0, 0
        current_step = 0
        if n_steps is None:
            n_steps = float("inf")

        # t
        # self.pre_rollout()
        # self.after_rollout()
        for agent_id, agent in self.policy_agents.items():
            agent.myoffer = None

        self.debug_actions = []

        self.pre_rollout()

        while any(running):
            # random.shuffle(indices)
            self.pre_rollout()
            self.broken = {agent_id: False for agent_id in self.env.trainable_agents}
            self.success = {agent_id: False for agent_id in self.env.trainable_agents}
            self.running = {agent_id: False for agent_id in self.env.trainable_agents}
            self.contract = {agent_id: None for agent_id in self.env.trainable_agents}

            for i in indices:
                if not running[i]:
                    continue
                if self.world.time >= self.world.time_limit:
                    self.tmp_terminated = True
                    break
                mechanism = mechanisms[i]
                agent_id = []
                if mechanism.negotiators[0].ami.annotation['seller'] in self.policy_agents:
                    _agent = mechanism.negotiators[0].ami.annotation['seller']
                    _index = self.policy_agents[_agent].awi.my_consumers.index(mechanism.negotiators[0].ami.annotation['buyer'])
                    agent_id.append(f"{_agent}_{_index}")

                if mechanism.negotiators[0].ami.annotation['buyer'] in self.policy_agents:
                    _agent = mechanism.negotiators[0].ami.annotation['buyer']
                    _index = self.policy_agents[_agent].awi.my_suppliers.index(mechanism.negotiators[0].ami.annotation['seller'])
                    agent_id = f"{_agent}_{_index}"

                contract, r = self.world._step_a_mechanism(mechanism, force_immediate_signing)
                contracts[i] = contract
                running[i] = r

                if not running[i]:
                    if contract is None:
                        for _ in agent_id:
                            self.broken[_] = True
                        n_broken_ += 1
                        n_steps_broken_ += mechanism.state.step + 1
                    else:
                        for _ in agent_id:
                            self.success[_] = True
                            self.contract[_] = contract
                        n_success_ += 1
                        n_steps_success_ += mechanism.state.step + 1
                    for _p in partners:
                        self.world._add_edges(
                            _p[0],
                            _p,
                            self.world._edges_negotiations_succeeded
                            if contract is not None
                            else self.world._edges_negotiations_failed,
                            issues=mechanism.issues,
                            bi=True,
                        )
                else:
                    for _ in agent_id:
                        self.running[_] = True
            current_step += 1
            if current_step >= n_steps:
                self.tmp_terminated = True
                break
            if self.world.time >= self.world.time_limit:
                self.tmp_terminated = True
                break

            self.after_rollout()

        return (
            contracts,
            running,
            n_steps_broken_,
            n_steps_success_,
            n_broken_,
            n_success_,
        )

    def run(self):
        # result = self.world.run(self._rl_runner)
        # result = self.world.run()
        """Runs the simulation until it ends"""
        self.world.train_world = self
        self.world._start_time = time.perf_counter()
        for _ in range(self.world.n_steps):
            if self.world.time >= self.world.time_limit:
                self.tmp_terminated = True
                break

            result = self.step()

            # reward after every negotiation finished
            negotiation_end_reward = []
            for agent_id, agent in self.policy_agents.items():
                negotiation_end_reward.append(self.world._profits[agent_id][-1])
            self.rollout_worker.tmp_r[-1] = [i + j for i, j in zip(self.rollout_worker.tmp_r[-1], negotiation_end_reward)]
            self.rollout_worker.tmp_episode_reward += sum(negotiation_end_reward)
            if not result:
                break

        self.tmp_terminated = True
        extra_reward = []
        for agent_id, score in self.world.scores().items():
            if agent_id in self.policy_agents:
                extra_reward.append(100*score)
            else:
                extra_reward.append(-100*score)
        self.rollout_worker.tmp_terminate[-1] = [self.tmp_terminated]

        # self.rollout_worker.tmp_r[-1][-1] += sum(extra_reward)
        # self.rollout_worker.tmp_episode_reward += sum(extra_reward)

        print(f"World Scores are: {self.world.scores()}")


    def pre_rollout(self):
        # A real training step
        # self.tmp_obs = self.env.get_obs()
        self.tmp_obs_dict = self.env.get_obs(type=dict)
        self.tmp_state = self.env.get_state()
        self.agent_idx = self.tmp_obs_dict.keys()
        # before execute action, need to reset these parameters
        self.tmp_actions, self.tmp_avail_actions, self.tmp_actions_onehot = [], [], []
        self.tmp_actions_dict = {agent: None for agent in self.env.trainable_agents}
        self.tmp_avail_actions_dict = {agent: None for agent in self.env.trainable_agents}
        self.tmp_actions_onehot_dict = {agent: None for agent in self.env.trainable_agents}
        self.set_agent = []
        self.tmp_terminated = False

    def after_rollout(self):
        # save something after one step mechanism
        # TOOD: set up the episode runner batch information after step mechianism
        # set the parameters in rl runner

        if None in list(self.tmp_actions_dict.values()):
            # raise ValueError("Actions are None, agents do not execute actions "
            #                  "No negotiations exist between agents!")
            # means this mechanism is finished
            # print(f"test: world_step: {self.world.current_step}, len_of_negotiation: {len(self.rollout_worker.tmp_u)}")
            # tmp_reward = self.rl_runner.env.get_reward()
            # self.rollout_worker.tmp_r[-1][-1] += tmp_reward
            # self.rollout_worker.tmp_episode_reward += tmp_reward
            # for _ in range(len(self.policy_agents)):
            #     action = 0
            #     action_onehot = np.zeros(self.rl_runner.args.n_actions)
            #     avail_action = [0] * self.rl_runner.args.n_actions
            #     self.tmp_actions.append(int(action))
            #     self.tmp_actions_onehot.append(action_onehot)
            #     self.tmp_avail_actions.append(avail_action)
            #     self.rollout_worker.tmp_last_action[_] = action_onehot

            for agent_id in self.env.trainable_agents:
                if self.tmp_actions_dict[agent_id] is None:
                    action = 0
                    action_onehot = np.zeros(self.rl_runner.args.n_actions)
                    avail_action = [0] * self.rl_runner.args.n_actions
                    self.tmp_actions_dict[agent_id] = int(action)
                    self.tmp_actions_onehot_dict[agent_id] = action_onehot
                    self.tmp_avail_actions_dict[agent_id] = avail_action
                    self.rollout_worker.tmp_last_action_dict[agent_id] = action_onehot

        self.debug_actions.append(list(self.tmp_actions_dict.values()))
        tmp_reward = self.rl_runner.env.get_reward()
        self.rollout_worker.tmp_o.append(list(self.tmp_obs_dict.values()))
        self.rollout_worker.tmp_s.append(self.tmp_state)
        try:
            self.rollout_worker.tmp_u.append(np.reshape(list(self.tmp_actions_dict.values()), [self.rollout_worker.n_agents, 1]))
            if any(list(self.contract.values())):
                partners = [c.partners for agent_id, c in self.contract.items() if c is not None]
                agreements = [c.agreement for agent_id, c in self.contract.items() if c is not None]
                print(f"Same propose: world_step:{self.world.current_step}, contract signed {partners}, agreements are {agreements}"
                      f"action is {self.debug_actions},"
                      f"world catalog price {self.world.info['catalog_prices']}")
        except Exception as e:
            # TODO: Enter here when Policy Agent negotiates with Policy Agent
            print(f"Accept: world_step:{self.world.current_step}, contract signed {self.contract}, action is {self.tmp_actions}")
            self.tmp_actions = self.tmp_actions * 2
            self.tmp_actions_onehot = self.tmp_actions_onehot * 2
            self.tmp_avail_actions = self.tmp_avail_actions * 2
            self.rollout_worker.tmp_last_action[1 - self.set_agent[0]] = self.rollout_worker.tmp_last_action[self.set_agent[0]]
            self.rollout_worker.tmp_u.append(np.reshape(self.tmp_actions, [self.rollout_worker.n_agents, 1]))
        self.rollout_worker.tmp_u_onehot.append(list(self.tmp_actions_onehot_dict.values()))
        self.rollout_worker.tmp_avail_u.append(list(self.tmp_avail_actions_dict.values()))
        self.rollout_worker.tmp_r.append([tmp_reward])
        self.rollout_worker.tmp_terminate.append([self.tmp_terminated])
        self.rollout_worker.tmp_padded.append([0.])
        self.rollout_worker.tmp_episode_reward += tmp_reward
        self.rollout_worker.tmp_step += 1
        if self.rollout_worker.args.epsilon_anneal_scale == 'step':
            self.rollout_worker.tmp_epsilon = self.rollout_worker.tmp_epsilon - self.rollout_worker.anneal_epsilon if \
                self.rollout_worker.tmp_epsilon > self.rollout_worker.min_epsilon else self.rollout_worker.tmp_epsilon

    def save_replay(self, replay_dir, prefix):
        pass



