"""Small JSON metadata contract; not an MRI or registration benchmark."""
import json
import numpy as np


def main():
    saved_shape = (np.int32(131), np.int32(241), np.int32(99))
    try:
        json.dumps({'actual_reloaded_shape':list(saved_shape)})
    except TypeError as error:
        assert str(error)=='Object of type int32 is not JSON serializable'
    else:
        raise AssertionError('the original serializer failure was not reproduced')
    explicit = [int(value) for value in saved_shape]
    assert explicit==[131,241,99]
    assert all(type(value) is int for value in explicit)
    assert json.loads(json.dumps({'actual_reloaded_shape':explicit}))['actual_reloaded_shape']==explicit
    print(json.dumps({'scope':'JSON object only; zero MRI/registration/sampling',
                      'old_int32_failure_reproduced':True,
                      'explicit_int_values_preserved':True,
                      'explicit_builtin_int_type':True,'JSON_roundtrip_exact':True,
                      'numpy_version':np.__version__},indent=2))


if __name__=='__main__':
    main()
