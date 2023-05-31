import logging
from datetime import datetime
from typing import Any, Iterator, List, Dict, Set, Mapping, Optional, Sequence, Tuple, Union
from typing_extensions import Self
import json
from bson import ObjectId

from pydantic import BaseModel
from pymongo.cursor import Cursor
from pymongo.results import BulkWriteResult, DeleteResult, UpdateResult
from pymongo.collection import Collection

from .connection import _get_client, get_db
from .data_conversion import dict2obj
from .exceptions import ObjectDoesNotExist
from .fields import Field
from .types import ODMObjectId  # type: ignore
from .utils.utils import convert_model_to_collection

logger = logging.getLogger(__name__)

INHERITANCE_FIELD_NAME = "_cls"
SORT_TYPE = Union[str, Sequence[Tuple[str, Union[int, str, Mapping[str, Any]]]]]


class _BaseDocument(BaseModel):
    class Config(BaseModel.Config):
        # Those fields will work as the default value of any child class.
        orm_mode: bool = True
        allow_population_by_field_name: bool = True
        collection_name: Optional[str] = None
        allow_inheritance: bool = False
        index_inheritance_field: bool = True

    def __setattr__(self, key, value) -> None:
        """Add '# type: ignore' as a comment if get type error while getting this value"""
        self.__dict__[key] = value

    @classmethod
    def _get_collection_class(cls) -> Any:
        model: Any = cls
        if model.__base__ != Document:
            base_model = model.__base__
            if (
                not hasattr(base_model.Config, "allow_inheritance")
                or base_model.Config.allow_inheritance is not True
            ):
                raise Exception(
                    f"Invalid model inheritance. {base_model} does not allow model inheritance."
                )
            if base_model.Config == model.Config:
                raise Exception(
                    f"Child Model{model.__name__} should declare a separate Config class."
                )
            return base_model, model
        else:
            return model, None

    @classmethod
    def _get_collection_name(cls) -> str:
        collection, _ = cls._get_collection_class()
        return convert_model_to_collection(collection)

    @classmethod
    def _get_child(cls) -> Optional[str]:
        _, collection = cls._get_collection_class()
        if collection is None:
            return None
        return convert_model_to_collection(collection)

    @classmethod
    def _get_collection(cls) -> Collection:
        if not hasattr(cls, "_collection") or cls._collection is None:
            db = get_db()
            cls._collection = db[cls._get_collection_name()]
        return cls._collection

    @classmethod
    def _db(cls) -> str:
        return cls._get_collection_name()

    @classmethod
    def get_inheritance_key(cls) -> dict:
        return {INHERITANCE_FIELD_NAME: cls._get_child()}

    @classmethod
    def start_session(cls):
        return _get_client().start_session()

    def __str__(self) -> str:
        return super().__repr__()

class SubDocument(_BaseDocument):
    """
    Placeholder for subdocument. This is needed so we don't generate and _id for subdocuments.
    """
    pass


class Document(_BaseDocument):
    _id: ODMObjectId = Field(default_factory=ObjectId)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if "_id" in kwargs:
            object.__setattr__(self, "_id", kwargs["_id"])
        else:
            object.__setattr__(self, "_id", self._id.default_factory())  # type: ignore

    def create(self, **kwargs) -> Self:
        """
        Create a new document in the database.

        :param kwargs: Additional arguments to pass to the insert_one method.
        :return: The document that was created.
        """
        _collection = self._get_collection()

        data = self.dict(exclude={"_id"}, exclude_none=True)
        if self._get_child() is not None:
            data = {**self.get_inheritance_key(), **data}

        inserted_id = _collection.insert_one(data, **kwargs).inserted_id
        self.__dict__.update({"_id": inserted_id})
        return self

    def to_json(self, exclude: Set[str] = None, use_aliases: bool = True) -> str:
        """
        Convert the document to a JSON string.

        It handles ObjectId and datetime objects.

        :param exclude: A set of fields to exclude from the JSON string.
        :param use_aliases: Whether to use aliases when serializing the document.
        :return: The JSON string.
        """
        data = self.dict(exclude=exclude, by_alias=use_aliases)
        serialized = {}
        for key, value in data.items():
            if isinstance(value, (ObjectId, datetime)):
                serialized[key] = str(value)
            elif isinstance(value, dict):
                serialized[key] = self.__class__(**value).to_json(exclude=exclude, use_aliases=use_aliases)
            else:
                serialized[key] = value
        return json.dumps(serialized)

    @classmethod
    def find_raw(cls, filter: Dict = None, projection: Dict = {}, **kwargs) -> Cursor:
        """
        Find all documents matching the filter. This method returns a raw cursor.

        :param filter: The filter to apply to the query.
        :param projection: The projection to apply to the query.
        :param kwargs: Additional arguments to pass to the find method.
        :return: A cursor over the documents matching the query.
        """
        if filter is None:
            filter = {}
        _collection = cls._get_collection()
        if cls._get_child() is not None:
            filter = {**cls.get_inheritance_key(), **filter}
        if projection:
            return _collection.find(filter, projection, **kwargs)
        return _collection.find(filter, **kwargs)

    @classmethod
    def find(
        cls,
        filter: Dict = None,
        sort: Optional[SORT_TYPE] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
        projection: Dict = {},
        **kwargs,
    ) -> Iterator[Self]:
        """
        Find all documents matching the filter.

        This method uses lazy evaluation, so the query is not executed until the
        returned cursor is iterated over.

        :param filter: The filter to apply to the query.
        :param sort: The sort order to apply to the query.
        :param skip: The number of documents to skip before returning.
        :param limit: The maximum number of documents to return.
        :param projection: The projection to apply to the query.
        :param kwargs: Additional arguments to pass to the find method.
        :return: A cursor over the documents matching the query.
        """
        if filter is None:
            filter = {}
        query_result = cls.find_raw(filter, projection, **kwargs)
        if sort:
            query_result = query_result.sort(sort)
        if skip:
            query_result = query_result.skip(skip)
        if limit:
            query_result = query_result.limit(limit)

        model_children = {}
        is_dynamic_model = False
        if (
            hasattr(cls.Config, "allow_inheritance")
            and cls.Config.allow_inheritance is True
        ):
            is_dynamic_model = True
            for model in cls.__subclasses__():
                model_children[cls._get_child()] = model

        for data in query_result:
            if is_dynamic_model and data.get(INHERITANCE_FIELD_NAME) in model_children:
                yield model_children[data[INHERITANCE_FIELD_NAME]](**data)
            else:
                yield cls(**data)

    @classmethod
    def find_one(
        cls, filter: Dict = None, sort: Optional[SORT_TYPE] = None, **kwargs
    ) -> Optional[Self]:
        """
        Get an object from the database, if it doesn't exist, return None.

        The difference between this method and get is that this method will return None if the
        object is not found.
        :param filter: The filter to use
        :param sort: The sort to use
        :param kwargs: Additional arguments to pass to the find_one method
        :return: The object or None
        """
        if filter is None:
            filter = {}
        query_result = cls.find_raw(filter, **kwargs)
        if sort:
            query_result = query_result.sort(sort)
        for data in query_result.limit(1):
            """limit 1 is equivalent to find_one and that is implemented in pymongo find_one"""
            return cls(**data)
        return None

    @classmethod
    def get(cls, filter: Dict, sort: Optional[SORT_TYPE] = None, **kwargs) -> Self:
        """
        Get an object from the database, if it doesn't exist, raise an exception.

        The difference between this method and find_one is that this method will raise an exception
        if the object is not found.
        :param filter: The filter to use
        :param sort: The sort to use
        :param kwargs: Additional arguments to pass to the find_one method
        :return: The object
        """
        document = cls.find_one(filter, sort, **kwargs)
        if document:
            return document
        raise ObjectDoesNotExist("Object not found.")

    @classmethod
    def get_or_create(
        cls, filter: Dict, sort: Optional[SORT_TYPE] = None, **kwargs
    ) -> Tuple[Self, bool]:
        """
        Try to get an object from the database, if it doesn't exist, create it.

        :param filter: The filter to use
        :param sort: The sort to use
        :param kwargs: Additional arguments to pass to the find_one method
        :return: A tuple of the object and a boolean indicating if it was created or not
        """
        document = cls.find_one(filter, sort, **kwargs)
        if document:
            return document, False
        return cls(**filter).create(), True

    @classmethod
    def count_documents(cls, filter: Dict = None, **kwargs) -> int:
        """
        Count the number of documents in the database matching the filter

        :param filter: The filter to use
        :param kwargs: Additional arguments to pass to the count_documents method
        :return: The number of documents matching the filter
        """
        if filter is None:
            filter = {}
        _collection = cls._get_collection()
        if cls._get_child() is not None:
            filter = {**cls.get_inheritance_key(), **filter}
        return _collection.count_documents(filter, **kwargs)

    @classmethod
    def exists(cls, filter: Dict = None, **kwargs) -> bool:
        """
        Check if a document exists in the database

        :param filter: The filter to use
        :param kwargs: Additional arguments to pass to the count_documents method
        :return: True if the document exists, False otherwise
        """
        return cls.count_documents(filter, **kwargs, limit=1) >= 1

    @classmethod
    def aggregate(
        cls, pipeline: List[Any], get_raw=False, inheritance_filter=True, **kwargs
    ) -> Iterator[Any]:
        """
        Use the aggregation framework to process documents.
        See https://pymongo.readthedocs.io/en/stable/examples/aggregation.html#aggregation-framework

        :param pipeline: A list of aggregation framework commands
        :param get_raw: If True, return the raw pymongo result
        :param inheritance_filter: If True, filter the results by the child class
        :param kwargs: Additional arguments to pass to the aggregate method
        :return: An iterator of the results
        """
        _collection = cls._get_collection()
        if inheritance_filter and cls._get_child() is not None:
            if len(pipeline) > 0 and "$match" in pipeline[0]:
                pipeline[0]["$match"] = {
                    f"{INHERITANCE_FIELD_NAME}": cls._get_child(),
                    **pipeline[0]["$match"],
                }
            else:
                pipeline = [
                    {"$match": {f"{INHERITANCE_FIELD_NAME}": cls._get_child()}}
                ] + pipeline
        for obj in _collection.aggregate(pipeline, **kwargs):
            if get_raw is True:
                yield obj
            else:
                yield dict2obj(obj)

    @classmethod
    def get_random_one(cls, filter: Dict = None, **kwargs) -> Self:
        """
        Get a random document from the collection.

        :param filter: Filter to apply to the query.
        :param kwargs: Additional arguments to pass to the aggregate method.
        :return: A random document.
        """
        if filter is None:
            filter = {}
        if cls._get_child() is not None:
            filter = {**cls.get_inheritance_key(), **filter}
        pipeline = [{"$match": filter}, {"$sample": {"size": 1}}]
        for data in cls.aggregate(pipeline, get_raw=True, **kwargs):
            return cls(**data)
        raise ObjectDoesNotExist("Object not found.")

    def update(self, raw: Dict = None, **kwargs) -> UpdateResult:
        """
        Update the current document.
        See https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.update_one

        :param raw: Raw data to update the document with.
        :param kwargs: Additional arguments to pass to the update_one method.
        :return: The result of the update.
        """
        if raw is None:
            raw = {}
        filter = {"_id": self._id}
        if raw:
            updated_data = raw
        else:
            updated_data = {"$set": self.dict(exclude={"_id", "id"}, exclude_none=True)}
        if hasattr(self, "updated_at"):
            datetime_now = datetime.utcnow()
            if "$set" not in updated_data:
                updated_data["$set"] = {}
            updated_data["$set"]["updated_at"] = datetime_now
            self.__dict__.update({"updated_at": datetime_now})

        return self.update_one(filter, updated_data, **kwargs)

    @classmethod
    def update_one(cls, filter: Dict = None, data: Dict = None, **kwargs) -> UpdateResult:
        """
        Update a single document matching the filter.
        See https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.update_one

        :param filter: A query that matches the document to update.
        :param data: The modifications to apply.
        :param kwargs: Optional arguments that ``update_one`` takes.
        :return: The result of the update.
        """
        if filter is None:
            filter = {}
        if data is None:
            data = {}

        _collection = cls._get_collection()
        if cls._get_child() is not None:
            filter = {**cls.get_inheritance_key(), **filter}
        return _collection.update_one(filter, data, **kwargs)

    @classmethod
    def update_many(cls, filter: Dict = None, data: Dict = None, **kwargs) -> UpdateResult:
        """
        Update all documents matching the filter.
        See https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.update_many

        :param filter: A query that matches the documents to update.
        :param kwargs: Optional arguments that ``update_many`` takes.
        :param data: The modifications to apply.
        """
        if filter is None:
            filter = {}
        if data is None:
            data = {}

        _collection = cls._get_collection()
        if cls._get_child() is not None:
            filter = {**cls.get_inheritance_key(), **filter}
        return _collection.update_many(filter, data, **kwargs)

    def delete(self, **kwargs) -> DeleteResult:
        return self.delete_one({"_id": self._id}, **kwargs)

    @classmethod
    def delete_one(cls, filter: Dict = None, **kwargs) -> DeleteResult:
        """
        Delete a single document matching the filter.
        See https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.delete_one

        :param filter: A query that matches the document to delete.
        :param kwargs: Optional arguments that ``delete_one`` takes.
        :return: The result of the delete operation.
        """
        if filter is None:
            filter = {}

        _collection = cls._get_collection()
        if cls._get_child() is not None:
            filter = {**cls.get_inheritance_key(), **filter}
        return _collection.delete_one(filter, **kwargs)

    @classmethod
    def delete_many(cls, filter: Dict = None, **kwargs) -> DeleteResult:
        """
        Delete all documents matching the filter.
        See https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.delete_many

        :param filter: A query that matches the documents to delete.
        :kwargs: Optional arguments that ``delete_many`` takes.
        :return: The result of the delete operation.
        """
        if filter is None:
            filter = {}

        _collection = cls._get_collection()
        if cls._get_child() is not None:
            filter = {**cls.get_inheritance_key(), **filter}
        return _collection.delete_many(filter, **kwargs)

    @classmethod
    def bulk_write(cls, requests: List, **kwargs) -> BulkWriteResult:
        """
        Perform a bulk write operation.
        See https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.bulk_write

        :param requests: A list of write operations (InsertOne, UpdateOne, UpdateMany, ReplaceOne, DeleteOne, DeleteMany)
        :param kwargs: Optional arguments that ``bulk_write`` takes.
        :return: The result of the bulk write operation
        """
        _collection = cls._get_collection()
        return _collection.bulk_write(requests, **kwargs)

    @classmethod
    def _traverse(cls, document: 'Document', keys: List[str], attr: str, value: Any) -> Optional[SubDocument]:
        """
        Recursively traverses a document object to find a subdocument
        :param document: The document object to traverse
        :param keys: The keys to traverse
        :param attr: The attribute to check
        :param value: The value to check
        :return: The subdocument, or None if not found
        """
        if len(keys) == 0:
            if hasattr(document, attr) and getattr(document, attr) == value:
                return document
            else:
                return None
        else:
            for key in keys:

                if document.__dict__[key] is None:
                    return None

                if isinstance(document, list) or isinstance(document.__dict__[key], list):
                    items = document if isinstance(document, list) else document.__dict__[key]
                    for item in items:
                        result = cls._traverse(item, keys[1:], attr, value)
                        if result is not None:
                            return result
                    return None
                else:
                    raise ValueError('The nested attribute must be a list of subdocuments')

    @classmethod
    def get_subdoc(cls, document: 'Document', filter_: Dict[str, Any]) -> Optional[SubDocument]:
        """
        Gets a nested subdocument from a document object
        The filter must be a dictionary with a single key, which must be a string in the format "nested.attribute", or
        "nested.nested.attribute". The nested attribute('s) must be a list of subdocuments.

        The value of the filter must be the value of the attribute to check.

        To store modifications to the subdocument, you must use the update method on the main document.

        Usage example:
        ```
        filter_ = {'subdoc_list.attribute': 'value'}
        main_doc = MyModel.get({'subdoc_list.attribute': 'value'})
        sub_doc = main_doc.get_subdoc(main_doc, filter_)
        sub_doc.attribute = 'new_value'
        main_doc.update()
        ```

        If you want to perform both a search operation and a modification operation on the subdocument, you can use the
        get_with_subdoc method instead.

        :param document: Document object to get the subdocument from
        :param filter_: Dictionary with a single key in the format "nested.attribute"
        :return: Subdocument object
        """
        if not isinstance(filter_, dict) or len(filter_) != 1:
            raise ValueError('Filter parameter must be a dictionary with a single key')
        filter_key = next(iter(filter_))
        *keys, attr = filter_key.split(".")
        value = filter_[filter_key]
        return cls._traverse(document, keys, attr, value)

    @classmethod
    def get_with_subdoc(cls, filter_: Dict[str, Any]) -> Tuple[Self, SubDocument]:
        """
        Find a document in the database using a filter on a value in a subdocument.
        The method will return the main document if the attribute is found in the subdocument, and also return
        the matched subdocument.

        The filter must be a dictionary with a single key, which must be a string in the format "nested.attribute", or
        "nested.nested.attribute". The nested attribute('s) must be a list of subdocuments.

        The value of the filter must be the value of the attribute to check.

        To store modifications to the subdocument, you must use the update method on the main document.

        Usage example:
        ```
        main_doc, sub_doc = MyModel.get_with_subdoc({'subdoc_list.attribute': 'value'})
        sub_doc.attribute = 'new_value'
        main_doc.update()
        ```
        :param filter_: The filter to use to find the document
        :return: The main document and the subdocument
        """

        document = cls.find_one(filter_)
        if document is None:
            return None, None
        object_searched = cls.get_subdoc(document, filter_)

        return document, object_searched

    def subdoc(self, filter_: Dict[str, Any]) -> Optional[SubDocument]:
        """
        Returns a nested subdocument from a document object
        The filter must be a dictionary with a single key, which must be a string in the format "nested.attribute", or
        "nested.nested.attribute". The nested attribute('s) must be a list of subdocuments.

        The value of the filter must be the value of the attribute to check.

        To store modifications to the subdocument, you must use the update method on the main document.

        Usage example:
        ```
        filter_ = {'subdoc_list.attribute': 'value'}
        main_doc = MyModel.get({'subdoc_list.attribute': 'value'})
        sub_doc = main_doc.subdoc(filter_)
        sub_doc.attribute = 'new_value'
        main_doc.update()
        ```

        :param filter_: The filter to use
        :return: The subdocument, or None if not found
        """
        return self.__class__.get_subdoc(self, filter_)
