"""
This file exists as the Python encoding of all Message types that Syft sends over the network. It is
an important bottleneck in the system, impacting both security, performance, and cross-platform
compatability. As such, message types should strive to not be framework specific (i.e., Torch,
Tensorflow, etc.).

All Syft message types extend the Message class.
"""

import syft as sy
from syft.workers.abstract import AbstractWorker

from syft.execution.computation import ComputationAction
from syft.frameworks.torch.tensors.interpreters.placeholder import PlaceHolder

from syft_proto.messaging.v1.message_pb2 import ObjectMessage as ObjectMessagePB
from syft_proto.messaging.v1.message_pb2 import CommandMessage as CommandMessagePB


class Message:
    """All syft message types extend this class

    All messages in the pysyft protocol extend this class. This abstraction
    requires that every message has an integer type, which is important because
    this integer is what determines how the message is handled when a BaseWorker
    receives it.

    Additionally, this type supports a default simplifier and detailer, which are
    important parts of PySyft's serialization and deserialization functionality.
    You can read more abouty detailers and simplifiers in syft/serde/serde.py.
    """

    def __init__(self, contents=None):

        # saves us a write op but costs us a check op to only sometimes
        # set ._contents
        if contents is not None:
            self._contents = contents

    @property
    def contents(self):
        """Return a tuple with the contents of the message (backwards compatability)

        Some of our codebase still assumes that all message types have a .contents attribute. However,
        the contents attribute is very opaque in that it doesn't put any constraints on what the contents
        might be. Some message types can be more efficient by storing their contents more explicitly (see
        CommandMessage). They can override this property to return a tuple view on their other properties.
        """
        if hasattr(self, "_contents"):
            return self._contents
        else:
            return None

    def _simplify(self):
        return (self.contents,)

    def __str__(self):
        """Return a human readable version of this message"""
        return f"({type(self).__name__} {self.contents})"

    def __repr__(self):
        """Return a human readable version of this message"""
        return self.__str__()

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "Message") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """

        return (sy.serde.msgpack.serde._simplify(worker, msg.contents),)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "Message":
        """
        This function takes the simplified tuple version of this message and converts
        it into a message. The simplify() method runs the inverse of this method.

        This method shouldn't get called very often. It exists as a backup but in theory
        every message type should have its own detailer.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            msg (Message): a Message.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """

        # TODO: attempt to use the msg_tuple[0] to return the correct type instead of Message
        # https://github.com/OpenMined/PySyft/issues/2514
        # TODO: as an alternative, this detailer could raise NotImplementedException
        # https://github.com/OpenMined/PySyft/issues/2514

        return Message(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))


class CommandMessage(Message):
    """All syft actions use this message type

    In Syft, an action is when one worker wishes to tell another worker to do something with
    objects contained in the worker._objects registry (or whatever the official object store is
    backed with in the case that it's been overridden). Semantically, one could view all Messages
    as a kind of action, but when we say action this is what we mean. For example, telling a
    worker to take two tensors and add them together is an action. However, sending an object
    from one worker to another is not an action (and would instead use the ObjectMessage type)."""

    def __init__(self, name, target, args_, kwargs_, return_ids):
        """Initialize an action message

        Args:
            message (Tuple): this is typically the args and kwargs of a method call on the client, but it
                can be any information necessary to execute the action properly.
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.), the id of
                action results are set by the client. This allows the client to be able to predict where
                the results will be ahead of time. Importantly, this allows the client to pre-initalize the
                pointers to the future data, regardless of whether the action has yet executed. It also
                reduces the size of the response from the action (which is very often empty).

        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.action = ComputationAction(name, target, args_, kwargs_, return_ids)

    @property
    def name(self):
        return self.action.name

    @property
    def target(self):
        return self.action.target

    @property
    def args(self):
        return self.action.args

    @property
    def kwargs(self):
        return self.action.kwargs

    @property
    def return_ids(self):
        return self.action.return_ids

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "CommandMessage") -> tuple:
        """
        This function takes the attributes of a CommandMessage and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (CommandMessage): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(ptr)
        """
        return (sy.serde.msgpack.serde._simplify(worker, ptr.action),)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "CommandMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into a CommandMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (CommandMessage): an CommandMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        action = msg_tuple[0]

        detailed = sy.serde.msgpack.serde._detail(worker, action)

        return CommandMessage(
            detailed.name, detailed.target, detailed.args, detailed.kwargs, detailed.return_ids
        )

    @staticmethod
    def bufferize(worker: AbstractWorker, action_message: "CommandMessage") -> "CommandMessagePB":
        """
        This function takes the attributes of a CommandMessage and saves them in Protobuf
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            action_message (CommandMessage): an CommandMessage
        Returns:
            protobuf_obj: a Protobuf message holding the unique attributes of the message
        Examples:
            data = bufferize(message)
        """
        protobuf_op_msg = CommandMessagePB()
        protobuf_op = ComputationAction.bufferize(worker, action_message.action)

        protobuf_op_msg.action.CopyFrom(protobuf_op)
        return protobuf_op_msg

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_obj: "CommandMessagePB") -> "CommandMessage":
        """
        This function takes the Protobuf version of this message and converts
        it into an CommandMessage. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (CommandMessagePB): the Protobuf message

        Returns:
            obj (CommandMessage): an CommandMessage

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """
        detailed = ComputationAction.unbufferize(worker, protobuf_obj.action)

        return CommandMessage(
            detailed.name, detailed.target, detailed.args, detailed.kwargs, detailed.return_ids
        )


class ObjectMessage(Message):
    """Send an object to another worker using this message type.

    When a worker has an object in its local object repository (such as a tensor) and it wants
    to send that object to another worker (and delete its local copy), it uses this message type
    to do so.
    """

    def __init__(self, object_):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__()

        self.object = object_

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "ObjectMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return (sy.serde.msgpack.serde._simplify(worker, msg.object),)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "ObjectMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an ObjectMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (ObjectMessage): a ObjectMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return ObjectMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))

    @staticmethod
    def bufferize(worker: AbstractWorker, message: "ObjectMessage") -> "ObjectMessagePB":
        """
        This function takes the attributes of an Object Message and saves them in a protobuf object
        Args:
            message (ObjectMessage): an ObjectMessage
        Returns:
            protobuf: a protobuf object holding the unique attributes of the object message
        Examples:
            data = bufferize(object_message)
        """

        protobuf_obj_msg = ObjectMessagePB()
        bufferized_obj = sy.serde.protobuf.serde._bufferize(worker, message.object)
        protobuf_obj_msg.tensor.CopyFrom(bufferized_obj)
        return protobuf_obj_msg

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_obj: "ObjectMessagePB") -> "ObjectMessage":
        protobuf_obj = protobuf_obj.tensor
        object_ = sy.serde.protobuf.serde._unbufferize(worker, protobuf_obj)
        object_msg = ObjectMessage(object_)

        return object_msg


class ObjectRequestMessage(Message):
    """Request another worker to send one of its objects

    If ObjectMessage pushes an object to another worker, this Message type pulls an
    object from another worker. It also assumes that the other worker will delete it's
    local copy of the object after sending it to you."""

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, obj_id, user, reason):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__()

        self.object_id = obj_id
        self.user = user
        self.reason = reason

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "ObjectRequestMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, msg.object_id),
            sy.serde.msgpack.serde._simplify(worker, msg.user),
            sy.serde.msgpack.serde._simplify(worker, msg.reason),
        )

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "ObjectRequestMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an ObjectRequestMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (ObjectRequestMessage): a ObjectRequestMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return ObjectRequestMessage(
            sy.serde.msgpack.serde._detail(worker, msg_tuple[0]),
            sy.serde.msgpack.serde._detail(worker, msg_tuple[1]),
            sy.serde.msgpack.serde._detail(worker, msg_tuple[2]),
        )


class IsNoneMessage(Message):
    """Check if a worker does not have an object with a specific id.

    Occasionally we need to verify whether or not a remote worker has a specific
    object. To do so, we send an IsNoneMessage, which returns True if the object
    (such as a tensor) does NOT exist."""

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, obj_id):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__()

        self.object_id = obj_id

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "IsNoneMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return (sy.serde.msgpack.serde._simplify(worker, msg.object_id),)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "IsNoneMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an IsNoneMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (IsNoneMessage): a IsNoneMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return IsNoneMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))


class GetShapeMessage(Message):
    """Get the shape property of a tensor in PyTorch

    We needed to have a special message type for this because .shape had some
    constraints in the older version of PyTorch."""

    # TODO: remove this message type and use ObjectRequestMessage instead.
    # note that the above to do is likely waiting for custom tensor type support in PyTorch
    # https://github.com/OpenMined/PySyft/issues/2513

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, tensor):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__()

        self.tensor = tensor

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "GetShapeMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return (sy.serde.msgpack.serde._simplify(worker, msg.tensor),)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "GetShapeMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an GetShapeMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            msg (GetShapeMessage): a GetShapeMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return GetShapeMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))


class ForceObjectDeleteMessage(Message):
    """Garbage collect a remote object

    This is the dominant message for garbage collection of remote objects. When
    a pointer is deleted, this message is triggered by default to tell the object
    being pointed to to also delete itself.
    """

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, obj_id):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__()

        self.object_id = obj_id

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "ForceObjectDeleteMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return (sy.serde.msgpack.serde._simplify(worker, msg.object_id),)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "ForceObjectDeleteMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an ForceObjectDeleteMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            msg (ForceObjectDeleteMessage): a ForceObjectDeleteMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return ForceObjectDeleteMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))


class SearchMessage(Message):
    """A client queries for a subset of the tensors on a remote worker using this type

    For some workers like SocketWorker we split a worker into a client and a server. For
    this configuration, a client can request to search for a subset of tensors on the server
    using this message type (this could also be called a "QueryMessage").
    """

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, query):
        """Initialize the message using default Message constructor.

        See Message.__init__ for details."""
        super().__init__()

        self.query = query

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "SearchMessage") -> tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return (sy.serde.msgpack.serde._simplify(worker, msg.query),)

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "SearchMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into an SearchMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (SearchMessage): a SearchMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return SearchMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))


class PlanCommandMessage(Message):
    """Message used to execute a command related to plans."""

    # TODO: add more efficient detailer and simplifier custom for this type
    # https://github.com/OpenMined/PySyft/issues/2512

    def __init__(self, command_name: str, args_: tuple):
        """Initialize a PlanCommandMessage.

        Args:
            command_name (str): name used to identify the command.
            message (Tuple): this is typically the args and kwargs of a method call on the client, but it
                can be any information necessary to execute the command properly.
        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.command_name = command_name
        self.args = args_

    @property
    def contents(self):
        """Returns a tuple with the contents of the action (backwards compatability)."""
        return (self.command_name, self.args)

    @staticmethod
    def simplify(worker: AbstractWorker, msg: "PlanCommandMessage") -> tuple:
        """
        This function takes the attributes of a PlanCommandMessage and saves them in a tuple

        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (PlanCommandMessage): a Message

        Returns:
            tuple: a tuple holding the unique attributes of the message
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, msg.command_name),
            sy.serde.msgpack.serde._simplify(worker, msg.args),
        )

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "PlanCommandMessage":
        """
        This function takes the simplified tuple version of this message and converts
        it into a PlanCommandMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (PlanCommandMessage): a PlanCommandMessage.
        """
        command_name, args_ = msg_tuple
        return PlanCommandMessage(
            sy.serde.msgpack.serde._detail(worker, command_name),
            sy.serde.msgpack.serde._detail(worker, args_),
        )
