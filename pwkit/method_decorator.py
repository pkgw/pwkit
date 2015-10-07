# Copied from the GitHub repository described below. A few modifications made.
# Override __call__ and possibly fixup

'''
Python decorator that knows the class the decorated method is bound to.

Please see full description here:
https://github.com/denis-ryzhkov/method_decorator/blob/master/README.md

method_decorator version 0.1.3
Copyright (C) 2013 by Denis Ryzhkov <denisr@denisr.com>
MIT License, see http://opensource.org/licenses/MIT
'''

__all__ = str ('method_decorator').split ()

#### method_decorator

class method_decorator(object):

    def __init__(self, func, obj=None, cls=None, method_type='function'):
        # These defaults are OK for plain functions and will be changed by
        # __get__() for methods once a method is dot-referenced.
        self.func, self.obj, self.cls, self.method_type = func, obj, cls, method_type

    def fixup (self, newobj):
        pass

    def __get__(self, obj=None, cls=None):
        # It is executed when decorated func is referenced as a method:
        # cls.func or obj.func.

        if self.obj == obj and self.cls == cls:
            return self # Use the same instance that is already processed by
                        # previous call to this __get__().

        method_type = (
            'staticmethod' if isinstance(self.func, staticmethod) else
            'classmethod' if isinstance(self.func, classmethod) else
            'instancemethod'
            # No branch for plain function - correct method_type for it is
            # already set in __init__() defaults.
        )

        # Use specialized method_decorator (or descendant) instance, don't
        # change current instance attributes - it leads to conflicts.
        newobj = object.__getattribute__(self, '__class__')(
            # Use bound or unbound method with this underlying func.
            self.func.__get__(obj, cls), obj, cls, method_type)
        self.fixup (newobj)
        return newobj

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getattribute__(self, attr_name): # Hiding traces of decoration.
        if attr_name in ('__init__', '__get__', '__call__', '__getattribute__',
                         'func', 'obj', 'cls', 'method_type', 'fixup'):
            # Our known names. '__class__' is not included because is used
            # only with explicit object.__getattribute__().
            return object.__getattribute__(self, attr_name) # Stopping recursion.

        # All other attr_names, including auto-defined by system in self, are
        # searched in decorated self.func, e.g.: __module__, __class__,
        # __name__, __doc__, im_*, func_*, etc. Raises correct AttributeError
        # if name is not found in decorated self.func.
        return getattr(self.func, attr_name)

    def __repr__(self):
        # Special case: __repr__ ignores __getattribute__.
        return self.func.__repr__()

#### test

def test():

    #### my_decorator

    class my_decorator(method_decorator):
        def __call__(self, *args, **kwargs):

            print('Calling {method_type} {method_name} from instance {instance} of class {class_name} from module {module_name} with args {args} and kwargs {kwargs}.'.format(
                method_type=self.method_type,
                method_name=self.__name__,
                instance=self.obj,
                class_name=(self.cls.__name__ if self.cls else None),
                module_name=self.__module__,
                args=args,
                kwargs=kwargs,
            ))

            return method_decorator.__call__(self, *args, **kwargs)

    #### MyClass

    class MyClass(object):

        @my_decorator
        def my_instance_method(self, arg, kwarg='default'):
            '''my_instance_method doc.'''
            return dict(arg=arg, kwarg=kwarg)

        @my_decorator
        @classmethod
        def my_class_method(cls, arg):
            return arg

        @my_decorator
        @staticmethod
        def my_static_method(arg):
            return arg

    my_class_module_name = MyClass.__module__
    my_instance = MyClass()

    #### my_plain_function

    @my_decorator
    def my_plain_function(arg):
        return arg

    #### instancemethod

    result = my_instance.my_instance_method
    assert result.method_type == 'instancemethod'
    assert result.__name__ == 'my_instance_method'
    assert result.obj == my_instance
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert result.__doc__ == 'my_instance_method doc.'
    assert result.im_self == my_instance
    assert result.im_class == MyClass
    assert repr(type(result.im_func)) == "<type 'function'>"
    assert result.func_defaults == ('default', )

    try:
        result.invalid
        assert False, 'Expected AttributeError'
    except AttributeError:
        pass

    result = my_instance.my_instance_method('bound', kwarg='kwarg')
    assert result == dict(arg='bound', kwarg='kwarg')

    result = my_instance.my_instance_method('bound')
    assert result == dict(arg='bound', kwarg='default')

    result = MyClass.my_instance_method
    assert result.method_type == 'instancemethod'
    assert result.__name__ == 'my_instance_method'
    assert result.obj == None
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert result.__doc__ == 'my_instance_method doc.'
    assert result.im_self == None
    assert result.im_class == MyClass
    assert repr(type(result.im_func)) == "<type 'function'>"
    assert result.func_defaults == ('default', )

    result = MyClass.my_instance_method(MyClass(), 'unbound')
    assert result['arg'] == 'unbound'

    #### classmethod

    result = MyClass.my_class_method
    assert result.method_type == 'classmethod'
    assert result.__name__ == 'my_class_method'
    assert result.obj == None
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert result.im_self == MyClass
    assert result.im_class == type
    assert repr(type(result.im_func)) == "<type 'function'>", type(result.im_func)

    result = MyClass.my_class_method('MyClass.my_class_method')
    assert result == 'MyClass.my_class_method'

    result = my_instance.my_class_method
    assert result.method_type == 'classmethod'
    assert result.__name__ == 'my_class_method'
    assert result.obj == my_instance
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert result.im_self == MyClass
    assert result.im_class == type
    assert repr(type(result.im_func)) == "<type 'function'>", type(result.im_func)

    result = my_instance.my_class_method('my_instance.my_class_method')
    assert result == 'my_instance.my_class_method'

    #### staticmethod

    result = MyClass.my_static_method
    assert result.method_type == 'staticmethod'
    assert result.__name__ == 'my_static_method'
    assert result.obj == None
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert not hasattr(result, 'im_self')
    assert not hasattr(result, 'im_class')
    assert not hasattr(result, 'im_func')

    result = MyClass.my_static_method('MyClass.my_static_method')
    assert result == 'MyClass.my_static_method'

    result = my_instance.my_static_method
    assert result.method_type == 'staticmethod'
    assert result.__name__ == 'my_static_method'
    assert result.obj == my_instance
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert not hasattr(result, 'im_self')
    assert not hasattr(result, 'im_class')
    assert not hasattr(result, 'im_func')

    result = my_instance.my_static_method('my_instance.my_static_method')
    assert result == 'my_instance.my_static_method'

    #### plain function

    result = my_plain_function
    assert result.method_type == 'function'
    assert result.__name__ == 'my_plain_function'
    assert result.obj == None
    assert result.cls == None
    assert result.__module__ == my_class_module_name
    assert not hasattr(result, 'im_self')
    assert not hasattr(result, 'im_class')
    assert not hasattr(result, 'im_func')

    result = my_plain_function('my_plain_function')
    assert result == 'my_plain_function'

    #### OK

    print('OK')

if __name__ == '__main__':
    test()
