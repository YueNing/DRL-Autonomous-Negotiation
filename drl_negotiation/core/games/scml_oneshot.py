import time
import copy
import sys
import traceback
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
            else:
                agents[id] = agent
        return agents

    def reset(self):
        """TODO: reset the world"""
        pass

    def t_step(self):
        self.world.step()

    def step(self) -> bool:
        """A single simulation step"""
        # self.world.step()
        if self.world._start_time is None or self.world._start_time < 0:
            self.world._start_time = time.perf_counter()
        if self.world.time >= self.world.time_limit:
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
                    break

        def _sign_contracts():
            self.world._process_unsigned()

        def _simulation_step():
            nonlocal stage
            try:
                self.world.simulation_step(stage)
                if self.world.time >= self.world.time_limit:
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
    ) -> Tuple[List[Contract], List[bool], int, int, int, int]:
        """TODO coding here, observation, stat, reward, action"""
        return self.world._step_negotiations(
            mechanisms,
            n_steps,
            force_immediate_signing,
            partners
        )

    def run(self):
        # result = self.world.run(self._rl_runner)
        # result = self.world.run()
        """Runs the simulation until it ends"""
        self.world._start_time = time.perf_counter()
        for _ in range(self.world.n_steps):
            if self.world.time >= self.world.time_limit:
                break
            if not self.step():
                break



