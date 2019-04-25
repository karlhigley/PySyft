import torch as th
import syft as sy

import pytest


def test_init(workers):

    # Initialization Test A: making sure sensitivity, max_vals, min_vals, and entities
    # are calculated correctly

    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    assert x.sensitivity == th.tensor([1])
    assert x.max_vals == th.tensor([1])
    assert x.min_vals == th.tensor([0])
    assert (x.entities == th.tensor([[1, 0, 0, 0]])).all()

    # ensure it's calculated correctly even when sensitivity is greater than 1
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[2, 0, 0, 0]])
        )
    )

    assert x.sensitivity == th.tensor([2])
    assert x.max_vals == th.tensor([2])
    assert x.min_vals == th.tensor([0])
    assert (x.entities == th.tensor([[1, 0, 0, 0]])).all()

    # test when multiple entities are contributing to a value
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    assert x.sensitivity == th.tensor([1])
    assert x.max_vals == th.tensor([2])
    assert x.min_vals == th.tensor([0])
    assert (x.entities == th.tensor([[1, 1, 0, 0]])).all()

    # test when min_ent_conts go negative
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, -1, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    assert x.sensitivity == th.tensor([2])
    assert x.max_vals == th.tensor([2])
    assert x.min_vals == th.tensor([-2])
    assert (x.entities == th.tensor([[1, 1, 0, 0]])).all()


def test_fail_init(workers):
    # Initialization Test B: test the initialization failure modes

    # test when min_ent_conts are greater than max_ent_conts
    with pytest.raises(AssertionError):

        x = (
            th.tensor([1])
            .int()
            .track_sensitivity(
                min_ent_conts=th.tensor([[1, 1, 0, 0]]), max_ent_conts=th.tensor([[-1, -1, 0, 0]])
            )
        )

    with pytest.raises(RuntimeError):
        # test when min_ent_conts don't match max_ent_conts
        x = (
            th.tensor([1])
            .int()
            .track_sensitivity(
                min_ent_conts=th.tensor([[-1, -1, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
            )
        )

    # test when min_ent_conts and max_ent_conts are missing an outer dimension
    with pytest.raises(
        sy.frameworks.torch.tensors.decorators.sensitivity.MissingEntitiesDimensionException
    ):

        x = (
            th.tensor([1])
            .int()
            .track_sensitivity(
                min_ent_conts=th.tensor([-1, -1, 0, 0]), max_ent_conts=th.tensor([1, 1, 0, 0])
            )
        )

    # test when a tensor's value is outside of the range specified by min_ent_conts and max_ent_conts
    with pytest.raises(
        sy.frameworks.torch.tensors.decorators.sensitivity.ValuesOutOfSpecifiedMinMaxRangeException
    ):

        # negative, non-positive, single entitiy, overlapping, symmetric add
        x = (
            th.tensor([1])
            .int()
            .track_sensitivity(
                min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[0, 0, 0, 0]])
            )
        )


def test_add():
    # Test Add

    # positive, non-negative, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x + x

    assert y.sensitivity == th.tensor([2])
    assert y.max_vals == th.tensor([2])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, non-positive, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[0, 0, 0, 0]])
        )
    )

    y = x + x

    assert y.sensitivity == th.tensor([2])
    assert y.max_vals == th.tensor([0])
    assert y.min_vals == th.tensor([-2])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x + x

    assert y.sensitivity == th.tensor([4])
    assert y.max_vals == th.tensor([2])
    assert y.min_vals == th.tensor([-2])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, dual entitiy, overlapping, symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = x + x

    assert y.sensitivity == th.tensor([4])
    assert y.max_vals == th.tensor([4])
    assert y.min_vals == th.tensor([-2])
    assert (y.entities == th.tensor([[1, 1, 0, 0]])).all()

    # negative, positive, dual entitiy, non-overlapping, non-symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = (
        th.tensor([5])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[4, 0, 0, 0]]), max_ent_conts=th.tensor([[5, 5, 0, 0]])
        )
    )

    z = x + y

    assert z.sensitivity == th.tensor([6])
    assert z.max_vals == th.tensor([12])
    assert z.min_vals == th.tensor([3])
    assert (z.entities == th.tensor([[1, 1, 0, 0]])).all()


def test_scalar_add():
    # positive, non-negative, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([1])
            .int()
            .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x + 1

    assert y.sensitivity == th.tensor([1])
    assert y.max_vals == th.tensor([2])
    assert y.min_vals == th.tensor([1])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, non-positive, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([-1])
            .int()
            .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[0, 0, 0, 0]])
        )
    )

    y = x + 1

    assert y.sensitivity == th.tensor([1])
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([-1])
            .int()
            .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x + 1

    assert y.sensitivity == th.tensor([2])
    assert y.max_vals == th.tensor([2])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, dual entitiy, overlapping, symmetric add
    x = (
        th.tensor([1])
            .int()
            .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = x + 1

    assert y.sensitivity == th.tensor([2])
    # assert y.max_vals == th.tensor([3]) #TODO: fix
    # assert y.min_vals == th.tensor([0]) #TODO: fix
    assert (y.entities == th.tensor([[1, 1, 0, 0]])).all()

def test_sub():
    # positive, non-negative, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x - x

    assert y.sensitivity == th.tensor([2])
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([-1])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, non-positive, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[0, 0, 0, 0]])
        )
    )

    y = x - x

    assert y.sensitivity == th.tensor([2])
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([-1])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x - x

    assert y.sensitivity == th.tensor([4])
    assert y.max_vals == th.tensor([2])
    assert y.min_vals == th.tensor([-2])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, dual entitiy, overlapping, symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = x - x

    assert y.sensitivity == th.tensor([4])
    assert y.max_vals == th.tensor([3])
    assert y.min_vals == th.tensor([-3])
    assert (y.entities == th.tensor([[1, 1, 0, 0]])).all()


def test_mult():

    # TODO: finish tests which test all failure modes

    # positive, non-negative, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x * x

    assert y.sensitivity == th.tensor([1])
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, non-positive, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[0, 0, 0, 0]])
        )
    )

    y = x * x

    assert y.sensitivity == th.tensor([1])
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x * x

    assert y.sensitivity == th.tensor([2])
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([-1])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, dual entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = x * x

    assert y.sensitivity == th.tensor([4])
    assert y.max_vals == th.tensor([4])
    assert y.min_vals == th.tensor([-3])
    assert (y.entities == th.tensor([[1, 1, 0, 0]])).all()

    # negative, positive, dual entitiy, non-overlapping, non-symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = (
        th.tensor([5])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[4, 0, 0, 0]]), max_ent_conts=th.tensor([[5, 5, 0, 0]])
        )
    )

    z = x * y

    assert z.sensitivity == th.tensor([20])
    assert z.max_vals == th.tensor([20])
    assert z.min_vals == th.tensor([-10])
    assert (z.entities == th.tensor([[1, 1, 0, 0]])).all()


def test_neg():
    # positive, non-negative, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = -x

    assert y.sensitivity == th.tensor([1])
    assert y.max_vals == th.tensor([0])
    assert y.min_vals == th.tensor([-1])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, non-positive, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[0, 0, 0, 0]])
        )
    )

    y = -x

    assert y.sensitivity == th.tensor([1])
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = -x

    assert y.sensitivity == th.tensor([2])
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([-1])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, dual entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = -x

    assert y.sensitivity == th.tensor([2])
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([-2])
    assert (y.entities == th.tensor([[1, 1, 0, 0]])).all()


def test_scalar_division():
    # positive, non-negative, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0.0, 0, 0, 0]]), max_ent_conts=th.tensor([[1.0, 0, 0, 0]])
        )
    )

    y = x / 2

    assert y.sensitivity == th.tensor([0.5])
    assert y.max_vals == th.tensor([0.5])
    assert y.min_vals == th.tensor([0.0])
    assert (y.entities == th.tensor([[1.0, 0, 0, 0]])).all()


def test_gt():
    # TODO: this is not comprehensive - we need lots more tests for the various error modes

    # positive, non-negative, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x > y

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # positive, non-negative, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x > x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # negative, non-positive, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[0, 0, 0, 0]])
        )
    )

    y = x > x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # negative, positive, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x > x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # negative, positive, dual entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = x > x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor(
        [2]
    )  # TODO: this should return 1 but we don't have smart enough logic for that yet
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 1, 0, 0]]).byte()).all()

    # negative, positive, dual entitiy, non-overlapping, non-symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = (
        th.tensor([5])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[4, 0, 0, 0]]), max_ent_conts=th.tensor([[5, 5, 0, 0]])
        )
    )

    z = x > y

    assert z.sensitivity == th.tensor([1]).byte()
    assert z.max_vals == th.tensor(
        [2]
    )  # TODO: this should return 1 but we don't have smart enough logic for that yet
    assert z.min_vals == th.tensor([0])
    assert (z.entities == th.tensor([[1, 1, 0, 0]]).byte()).all()


def test_lt():
    # TODO: this is not comprehensive - we need lots more tests for the various error modes

    # positive, non-negative, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x < y

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # positive, non-negative, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x < x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # negative, non-positive, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[0, 0, 0, 0]])
        )
    )

    y = x < x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # negative, positive, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x < x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # negative, positive, dual entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = x < x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor(
        [2]
    )  # TODO: this should return 1 but we don't have smart enough logic for that yet
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 1, 0, 0]]).byte()).all()

    # negative, positive, dual entitiy, non-overlapping, non-symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = (
        th.tensor([5])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[4, 0, 0, 0]]), max_ent_conts=th.tensor([[5, 5, 0, 0]])
        )
    )

    z = x < y

    assert z.sensitivity == th.tensor([1]).byte()
    assert z.max_vals == th.tensor(
        [2]
    )  # TODO: this should return 1 but we don't have smart enough logic for that yet
    assert z.min_vals == th.tensor([0])
    assert (z.entities == th.tensor([[1, 1, 0, 0]]).byte()).all()


def test_eq():
    # positive, non-negative, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x == y

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # positive, non-negative, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x == x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # negative, non-positive, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[0, 0, 0, 0]])
        )
    )

    y = x == x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # negative, positive, single entitiy, overlapping, symmetric mult
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x == x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor([1])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]]).byte()).all()

    # negative, positive, dual entitiy, overlapping, symmetric mult
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = x == x

    assert y.sensitivity == th.tensor([1]).byte()
    assert y.max_vals == th.tensor(
        [2]
    )  # TODO: this should return 1 but we don't have smart enough logic for that yet
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 1, 0, 0]]).byte()).all()

    # negative, positive, dual entitiy, non-overlapping, non-symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = (
        th.tensor([5])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[4, 0, 0, 0]]), max_ent_conts=th.tensor([[5, 5, 0, 0]])
        )
    )

    z = x == y

    assert z.sensitivity == th.tensor([1]).byte()
    assert z.max_vals == th.tensor(
        [2]
    )  # TODO: this should return 1 but we don't have smart enough logic for that yet
    assert z.min_vals == th.tensor([0])
    assert (z.entities == th.tensor([[1, 1, 0, 0]]).byte()).all()
