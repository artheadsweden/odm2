from typing import Any, Optional, Union, AbstractSet, Mapping, TypeVar, Generic, Callable
from operator import itemgetter
from pydantic.fields import FieldInfo as PydanticFieldInfo, Undefined
from pydantic.typing import NoArgAnyCallable
from pydantic.utils import Representation
from pymongo import ReturnDocument
from svs_mongodb_odm.connection import get_db


IntStr = Union[int, str]
AbstractSetIntStr = AbstractSet[IntStr]
MappingIntStrAny = Mapping[IntStr, Any]


class FieldInfo(PydanticFieldInfo):
    def __init__(self, default: Any = Undefined, **kwargs: Any) -> None:
        super().__init__(default=default, **kwargs)


T = TypeVar("T")

class ListField(Generic[T]):
    """
    Custom field for storing a list of items in a MongoDB document and limiting the size of the list.

    :param max_length: The maximum number of items to store in the list. If the list exceeds this length, the oldest item will be removed.
    :param sort_key: The key to use for sorting the list. If this is set, the list will be sorted by this key before removing the oldest item.
    :param reversed: If True, the list will be sorted in reverse order before removing the oldest item.
    """
    def __init__(self, max_length: Optional[int] = None, sort_key: Optional[str] = None, reversed: bool = False) -> None:
        self.max_length = max_length
        self.sort_key = sort_key
        self.reversed = reversed
        self._list = []

    def append(self, item: T) -> None:
        """
        Append an item to the list.

        :param item: The item to append.
        """
        self._list.append(item)
        if self.max_length and len(self._list) > self.max_length:
            if self.sort_key:
                sorted_list = sorted(self._list, key=lambda x: getattr(x, self.sort_key))
                item_to_remove = sorted_list[0]
                self._list.remove(item_to_remove)
            else:
                self._list.pop(0)


    @classmethod
    def __get_validators__(cls) -> Callable[..., Any]:
        """
        Yield the validator for this field.

        :return: The validator for this field.
        """
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> "ListField":
        """
        Validate the value for this field.

        :param v: The value to validate.
        :return: The validated value.
        """
        if not isinstance(v, cls):
            raise TypeError("ListField expected")
        return cls(v)

    def __repr__(self) -> str:
        """
        Return a string representation of this field.
        """
        return repr(self._list)
    
    def __iter__(self) -> Any:
        """
        Return an iterator for this field.
        """
        return iter(self._list)

def Field(
    default: Any = Undefined,
    *,
    default_factory: Optional[NoArgAnyCallable] = None,
    alias: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    exclude: Union["AbstractSetIntStr", "MappingIntStrAny", Any] = None,
    include: Union["AbstractSetIntStr", "MappingIntStrAny", Any] = None,
    const: Optional[bool] = None,
    gt: Optional[float] = None,
    ge: Optional[float] = None,
    lt: Optional[float] = None,
    le: Optional[float] = None,
    multiple_of: Optional[float] = None,
    allow_inf_nan: Optional[bool] = None,
    max_digits: Optional[int] = None,
    decimal_places: Optional[int] = None,
    min_items: Optional[int] = None,
    max_items: Optional[int] = None,
    unique_items: Optional[bool] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_mutation: bool = True,
    regex: Optional[str] = None,
    discriminator: Optional[str] = None,
    repr: bool = True,
    **extra: Any,
) -> Any:
    field_info = FieldInfo(
        default,
        default_factory=default_factory,
        alias=alias,
        title=title,
        description=description,
        exclude=exclude,
        include=include,
        const=const,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        allow_inf_nan=allow_inf_nan,
        max_digits=max_digits,
        decimal_places=decimal_places,
        min_items=min_items,
        max_items=max_items,
        unique_items=unique_items,
        min_length=min_length,
        max_length=max_length,
        allow_mutation=allow_mutation,
        regex=regex,
        discriminator=discriminator,
        repr=repr,
        **extra,
    )
    field_info._validate()
    return field_info


def CounterField(
    collection: str,
    counter_name: str,
    **kwargs: Any) -> Any:

    """
    Field that generates a unique integer value for each document.
    The value is generated using a counter collection in the database.

    You can use create_counters_collection() function, found in svs_mongodb_odm.mongo_utils 
    to create the counter collection.

    :param collection: Name of the collection
    :param counter_name: Name of the counter

    :return: Field
    """

    def get_default() -> int:
        db = get_db()
        doc = db[collection].find_one_and_update(
            {"_id": counter_name},
            {"$inc": {"sequence_value": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return doc['sequence_value']

    return Field(default_factory=get_default, **kwargs)

class RelationshipInfo(Representation):
    def __init__(
        self,
        *,
        related_field: Optional[str] = None,
    ) -> None:
        self.related_field = related_field


def Relationship(
    *,
    related_field: Optional[str] = None,
) -> Any:
    relationship_info = RelationshipInfo(
        related_field=related_field,
    )
    return relationship_info
