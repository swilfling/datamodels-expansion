import sys


class BasicInterface:

    @classmethod
    def cls_from_name(cls, cls_type: str):
        """
        Get type of class from string (subclass of instance)
        Returns None if non-existent.
        @param cls_type: class name
        @return: class type
        """
        for mod in sys.modules.values():
            if hasattr(mod, cls_type):
                return getattr(mod, cls_type)
        return None

    @classmethod
    def from_name(cls, cls_type: str, **kwargs):
        """
        Get instance of class from string (subclass of interface)
        Returns None if non-existent.
        @param cls_type: class name
        @return: instance
        """
        return cls.cls_from_name(cls_type)(**kwargs) if cls.cls_from_name(cls_type) is not None else None

    def _set_attrs(self, **kwargs):
        """
        Set attributes of class - override if necessary
        """
        for name, val in kwargs.items():
            setattr(self, name, val)

    def _get_attrs(self):
        """
        Get attributes of class - override if necessary
        """
        return self.__dict__

