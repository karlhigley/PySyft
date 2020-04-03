import syft
from syft.generic.frameworks.hook import hook_args
from syft.execution.placeholder_id import PlaceholderId
from syft.generic.tensor import AbstractTensor
from syft.workers.abstract import AbstractWorker
from syft_proto.execution.v1.placeholder_pb2 import Placeholder as PlaceholderPB


class PlaceHolder(AbstractTensor):
    def __init__(self, owner=None, id=None, tags: set = None, description: str = None):
        """A PlaceHolder acts as a tensor but does nothing special. It can get
        "instantiated" when a real tensor is appended as a child attribute. It
        will send forward all the commands it receives to its child tensor.

        When you send a PlaceHolder, you don't sent the instantiated tensors.

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the PlaceHolder.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)

        if not isinstance(self.id, PlaceholderId):
            self.id = PlaceholderId(self.id)
        self.child = None

    # TODO do we remove the tracing implementation if we trace plans here?

    # def __getattribute__(self, name):
    #     """ This method is called each time we want to access an attribute or call a method
    #     of self.
    #     We implement our tracing logic here.
    #     """
    #     attribute = object.__getattribute__(self, name)
    #     if callable(attribute):
    #         # Trace
    #         print("tracing", attribute.__name__)
    #     return attribute

    def __add__(self, other):
        print("tracing add")
        # command = ('__add__', self.id, other.id, {})
        ph = PlaceHolder()
        res = self.child + other.child
        return ph.instantiate(res)

    @classmethod
    def handle_func_command(cls, command):
        """ Receive an instruction for a function to be applied on a Placeholder,
        Replace in the args with their child attribute, forward the command
        instruction to the handle_function_command of the type of the child attributes,
        get the response and wrap it in a Placeholder.
        We use this method to perform the tracing.

        Args:
            command: instruction of a function command: (command name,
                <no self>, arguments[, kwargs])

        Returns:
            the response of the function command
        """
        print("tracing", command)
        cmd, _, args, kwargs = command

        # Replace all FixedPrecisionTensor with their child attribute
        new_args, new_kwargs, new_type = hook_args.unwrap_args_from_function(cmd, args, kwargs)

        # build the new command
        new_command = (cmd, None, new_args, new_kwargs)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back FixedPrecisionTensor on the tensors found in the response
        result = PlaceHolder()
        response = result.instantiate(response)

        return response

    def instantiate(self, tensor):
        """
        Add a tensor as a child attribute. All operations on the placeholder will be also
        executed on this child tensor.

        We remove Placeholders if is there are any.
        """
        if isinstance(tensor, PlaceHolder):
            self.child = tensor.child
        else:
            self.child = tensor
        return self

    def __str__(self) -> str:
        """
        Compact representation of a Placeholder, including tags and optional child
        """
        tags = " ".join(list(self.tags or []))

        out = f"{type(self).__name__ }[Tags:{tags}]"

        if hasattr(self, "child") and self.child is not None:
            out += f">{self.child}"

        return out

    __repr__ = __str__

    def copy(self):
        """
        Copying a placeholder doesn't duplicate the child attribute, because all
        copy operations happen locally where we want to keep reference to the same
        instantiated object. As the child doesn't get sent, this is not an issue.
        """
        placeholder = PlaceHolder(tags=self.tags, owner=self.owner)
        placeholder.child = self.child
        return placeholder

    @staticmethod
    def create_placeholders(args_shape):
        """ Helper method to create a list of placeholders with shapes
        in args_shape.
        """
        # In order to support -1 value in shape to indicate any dimension
        # we map -1 to 1 for shape dimensions.
        # TODO: A more complex strategy could be used
        mapped_shapes = []
        for shape in args_shape:
            if list(filter(lambda x: x < -1, shape)):
                raise ValueError(f"Invalid shape {shape}")
            mapped_shapes.append(tuple(map(lambda y: 1 if y == -1 else y, shape)))

        return [syft.framework.hook.create_zeros(shape) for shape in mapped_shapes]

    @staticmethod
    def instantiate_placeholders(obj, response):
        """
        Utility function to instantiate recursively an object containing placeholders with a similar object but containing tensors
        """
        if obj is not None:
            if isinstance(obj, PlaceHolder):
                obj.instantiate(response)
            elif isinstance(obj, (list, tuple)):
                for ph, rep in zip(obj, response):
                    PlaceHolder.instantiate_placeholders(ph, rep)
            else:
                raise ValueError(
                    f"Response of type {type(response)} is not supported in Placeholder.instantiate."
                )

    @staticmethod
    def simplify(worker: AbstractWorker, tensor: "PlaceHolder") -> tuple:
        """Takes the attributes of a PlaceHolder and saves them in a tuple.

        Args:
            worker: the worker doing the serialization
            tensor: a PlaceHolder.

        Returns:
            tuple: a tuple holding the unique attributes of the PlaceHolder.
        """

        return (
            syft.serde.msgpack.serde._simplify(worker, tensor.id),
            syft.serde.msgpack.serde._simplify(worker, tensor.tags),
            syft.serde.msgpack.serde._simplify(worker, tensor.description),
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "PlaceHolder":
        """
            This function reconstructs a PlaceHolder given it's attributes in form of a tuple.
            Args:
                worker: the worker doing the deserialization
                tensor_tuple: a tuple holding the attributes of the PlaceHolder
            Returns:
                PlaceHolder: a PlaceHolder
            """

        tensor_id, tags, description = tensor_tuple

        tensor_id = syft.serde.msgpack.serde._detail(worker, tensor_id)
        tags = syft.serde.msgpack.serde._detail(worker, tags)
        description = syft.serde.msgpack.serde._detail(worker, description)

        return PlaceHolder(owner=worker, id=tensor_id, tags=tags, description=description)

    @staticmethod
    def bufferize(worker: AbstractWorker, tensor: "PlaceHolder") -> PlaceholderPB:
        """Takes the attributes of a PlaceHolder and saves them in a Protobuf message.

        Args:
            worker: the worker doing the serialization
            tensor: a PlaceHolder.

        Returns:
            PlaceholderPB: a Protobuf message holding the unique attributes of the PlaceHolder.
        """

        protobuf_placeholder = PlaceholderPB()
        syft.serde.protobuf.proto.set_protobuf_id(protobuf_placeholder.id, tensor.id.value)
        protobuf_placeholder.tags.extend(tensor.tags)

        if tensor.description:
            protobuf_placeholder.description = tensor.description

        return protobuf_placeholder

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_placeholder: PlaceholderPB) -> "PlaceHolder":
        """
            This function reconstructs a PlaceHolder given it's attributes in form of a Protobuf message.
            Args:
                worker: the worker doing the deserialization
                protobuf_placeholder: a Protobuf message holding the attributes of the PlaceHolder
            Returns:
                PlaceHolder: a PlaceHolder
            """

        tensor_id = syft.serde.protobuf.proto.get_protobuf_id(protobuf_placeholder.id)
        tags = set(protobuf_placeholder.tags)

        description = None
        if bool(protobuf_placeholder.description):
            description = protobuf_placeholder.description

        return PlaceHolder(owner=worker, id=tensor_id, tags=tags, description=description)


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PlaceHolder)
