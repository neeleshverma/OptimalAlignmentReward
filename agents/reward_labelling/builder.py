import functools
from typing import Callable, Iterator, List, Optional, Sequence

from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.jax.imitation_learning_types import DirectPolicyNetwork
from acme.jax.imitation_learning_types import DirectRLNetworks
from acme.utils import counting
from acme.utils import loggers
import reverb

from agents.reward_labelling import adder as ot_adder
from agents.reward_labelling import rewarder


class OptimalTransportBuilder(builders.ActorLearnerBuilder[DirectRLNetworks,
                                               DirectPolicyNetwork,
                                               reverb.ReplaySample]):
  """OptimalTransport Agent builder."""

  def __init__(
      self,
      rl_agent: builders.GenericActorLearnerBuilder[DirectRLNetworks,
                                                    DirectPolicyNetwork,
                                                    reverb.ReplaySample],
      make_demonstrations: Callable[[], Sequence[Sequence[types.Transition]]],
      episode_length: int,
      encoder_fn: Optional[rewarder.EncoderFn] = None,
      reward_scale: float = 10.0,
      preprocessor_update_period: int = 1,
  ):
    self._rl_agent = rl_agent
    self._make_demonstrations = make_demonstrations
    self._encoder_fn = encoder_fn
    self._reward_scale = reward_scale
    self._preprocessor_update_period = preprocessor_update_period
    self._episode_length = episode_length

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: DirectPolicyNetwork,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    return self._rl_agent.make_learner(
        random_key=random_key,
        networks=networks,
        dataset=dataset,
        logger_fn=logger_fn,
        environment_spec=environment_spec,
        replay_client=replay_client,
        counter=counter,
    )

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: DirectPolicyNetwork,
  ) -> List[reverb.Table]:
    return self._rl_agent.make_replay_tables(environment_spec, policy)

  def make_dataset_iterator(
      self,
      replay_client: reverb.Client) -> Optional[Iterator[reverb.ReplaySample]]:
    return self._rl_agent.make_dataset_iterator(replay_client)

  def make_adder(
      self,
      replay_client: reverb.Client,
      environment_spec: Optional[specs.EnvironmentSpec],
      policy: Optional[DirectPolicyNetwork],
  ) -> Optional[adders.Adder]:
    return self._rl_agent.make_adder(replay_client, environment_spec, policy)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: DirectPolicyNetwork,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> core.Actor:
    assert variable_source is not None
    if adder is not None:
      # If adder is given, then it's the adder from the direct rl agent.
      # Wrap it with the OptimalTransport rewarder
      if self._encoder_fn is not None:
        preprocessor = rewarder.EncoderPreprocessor(self._encoder_fn)
        variable_client = variable_utils.VariableClient(variable_source,
                                                        "policy")
        variable_client.update_and_wait()
      else:
        preprocessor = rewarder.MeanStdPreprocessor(
            environment_spec.observations,
            partial_update=False,
        )
        variable_client = None

      squashing_fn = functools.partial(
          rewarder.squashing_linear, alpha=self._reward_scale)

      ot_rewarder = rewarder.OptimalTransportRewarder(
          self._make_demonstrations(),
          episode_length=self._episode_length,
          preprocessor=preprocessor,
          squashing_fn=squashing_fn,
          preprocessor_update_period=self._preprocessor_update_period,
          variable_client=variable_client)

      adder = ot_adder.OptimalTransportAdder(
          direct_rl_adder=adder, ot_rewarder=ot_rewarder)

    return self._rl_agent.make_actor(
        random_key=random_key,
        policy=policy,
        environment_spec=environment_spec,
        adder=adder,
        variable_source=variable_source)
