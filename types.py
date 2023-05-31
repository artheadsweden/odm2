from bson import ObjectId


class ODMObjectId(ObjectId):
    """
    Custom type for ObjectId with basic validation
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        elif isinstance(v, str):
            return ObjectId(v)
        raise TypeError("Invalid data. ObjectId required")
