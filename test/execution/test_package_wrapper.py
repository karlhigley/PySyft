import syft as sy
import torch

from syft.execution.plan import trace
from syft.generic.frameworks import framework_packages


def test_plan_built_automatically():
    @sy.func2plan(args_shape=[(1,)])
    def plan_test(unused, torch=torch):
        y = torch.rand([1])
        return y

    p = plan_test(torch.tensor([3]))
    assert len(plan_test.role.actions) > 0
