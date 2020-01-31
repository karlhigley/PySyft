from collections import OrderedDict
import pytest
import numpy
import torch
from functools import partial
import traceback
import io

import syft
from syft.serde import protobuf
from test.serde.serde_helpers import *

# Dictionary containing test samples functions
samples = OrderedDict()

# Python Native
samples[type(None)] = make_none
# Other native Python types are not supported with Protobuf

# Numpy
# Not supported with Protobuf

# PyTorch
samples[torch.device] = make_torch_device
samples[torch.jit.ScriptModule] = make_torch_scriptmodule
samples[torch.jit.ScriptFunction] = make_torch_scriptfunction
samples[torch.jit.TopLevelTracedModule] = make_torch_topleveltracedmodule
samples[torch.nn.Parameter] = make_torch_parameter
samples[torch.Tensor] = make_torch_tensor
samples[torch.Size] = make_torch_size
# samples[torch.memory_format] = make_torch_memoryformat

# PySyft
# samples[syft.federated.train_config.TrainConfig] = make_trainconfig
# samples[syft.generic.string.String] = make_string
# samples[syft.workers.base.BaseWorker] = make_baseworker
# Dynamically added to msgpack.serde.simplifiers by some other test
# samples[syft.workers.virtual.VirtualWorker] = make_baseworker

# PySyft Pointers and Wrappers
# samples[syft.generic.pointers.multi_pointer.MultiPointerTensor] = make_multipointertensor
# samples[syft.generic.pointers.object_wrapper.ObjectWrapper] = make_objectwrapper
# samples[syft.generic.pointers.object_pointer.ObjectPointer] = make_objectpointer
samples[syft.generic.pointers.pointer_tensor.PointerTensor] = make_pointertensor
# samples[syft.generic.pointers.pointer_plan.PointerPlan] = make_pointerplan
# samples[syft.generic.pointers.pointer_protocol.PointerProtocol] = make_pointerprotocol

# PySyft Torch Tensor Interpreters and Decorators
# samples[syft.frameworks.torch.tensors.decorators.logging.LoggingTensor] = make_loggingtensor
samples[
    syft.frameworks.torch.tensors.interpreters.additive_shared.AdditiveSharingTensor
] = make_additivesharingtensor
# samples[syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor] = make_autogradtensor
# samples[
#     syft.frameworks.torch.tensors.interpreters.crt_precision.CRTPrecisionTensor
# ] = make_crtprecisiontensor
# samples[
#     syft.frameworks.torch.tensors.interpreters.precision.FixedPrecisionTensor
# ] = make_fixedprecisiontensor
# samples[syft.frameworks.torch.tensors.interpreters.gradients_core.GradFunc] = make_gradfn
# samples[syft.frameworks.torch.tensors.interpreters.private.PrivateTensor] = make_privatetensor
samples[syft.frameworks.torch.tensors.interpreters.placeholder.PlaceHolder] = make_placeholder

# PySyft Messaging
samples[syft.messaging.plan.plan.Plan] = make_plan
samples[syft.messaging.plan.state.State] = make_state
samples[syft.messaging.protocol.Protocol] = make_protocol

# PySyft Messages
# samples[syft.messaging.message.Message] = make_message
samples[syft.messaging.message.Operation] = make_operation
samples[syft.messaging.message.ObjectMessage] = make_objectmessage
# samples[syft.messaging.message.ObjectRequestMessage] = make_objectrequestmessage
# samples[syft.messaging.message.IsNoneMessage] = make_isnonemessage
# samples[syft.messaging.message.GetShapeMessage] = make_getshapemessage
# samples[syft.messaging.message.ForceObjectDeleteMessage] = make_forceobjectdeletemessage
# samples[syft.messaging.message.SearchMessage] = make_searchmessage
# samples[syft.messaging.message.PlanCommandMessage] = make_plancommandmessage

# PySyft Exceptions
# samples[syft.exceptions.GetNotPermittedError] = make_getnotpermittederror
# samples[syft.exceptions.ResponseSignatureError] = make_responsesignatureerror


def test_serde_coverage():
    """Checks all types in serde are tested"""
    for cls, _ in protobuf.serde.bufferizers.items():
        has_sample = cls in samples
        assert has_sample is True, "Serde for %s is not tested" % cls


@pytest.mark.parametrize("cls", samples)
def test_serde_roundtrip_protobuf(cls, workers):
    """Checks that values passed through serialization-deserialization stay same"""
    serde_worker = syft.hook.local_worker
    original_framework = serde_worker.framework
    _samples = samples[cls](workers=workers)
    for sample in _samples:
        _to_protobuf = (
            protobuf.serde._bufferize
            if not sample.get("forced", False)
            else protobuf.serde._force_full_bufferize
        )
        serde_worker.framework = sample.get("framework", torch)
        obj = sample.get("value")
        protobuf_obj = _to_protobuf(serde_worker, obj)
        roundtrip_obj = None
        if not isinstance(obj, Exception):
            roundtrip_obj = protobuf.serde._unbufferize(serde_worker, protobuf_obj)

        serde_worker.framework = original_framework

        if sample.get("cmp_detailed", None):
            # Custom detailed objects comparison function.
            assert sample.get("cmp_detailed")(roundtrip_obj, obj) is True
        else:
            assert type(roundtrip_obj) == type(obj)
            assert roundtrip_obj == obj
