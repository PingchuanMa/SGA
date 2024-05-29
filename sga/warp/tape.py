from typing import Optional
import warp as wp


class Tape(wp.Tape):

    # returns the adjoint of a kernel parameter
    def get_adjoint(self, a):
        if not wp.types.is_array(a) and not isinstance(a, wp.codegen.StructInstance):
            # if input is a simple type (e.g.: float, vec3, etc) then
            # no gradient needed (we only return gradients through arrays and structs)
            return a

        elif wp.types.is_array(a) and a.grad:
            # keep track of all gradients used by the tape (for zeroing)
            # ignore the scalar loss since we don't want to clear its grad
            self.gradients[a] = a.grad
            return a.grad

        elif isinstance(a, wp.codegen.StructInstance):
            adj = a._cls()
            for name, _ in a._cls.ctype._fields_:
                if name.startswith("_"):
                    continue
                if isinstance(a._cls.vars[name].type, wp.array):
                    arr = getattr(a, name)
                    if arr.grad:
                        grad = self.gradients[arr] = arr.grad
                    else:
                        grad = None
                    setattr(adj, name, grad)
                else:
                    setattr(adj, name, getattr(a, name))

            self.gradients[a] = adj
            return adj

        return None


class CondTape(object):
    def __init__(self, tape: Optional[Tape], cond: bool = True) -> None:
        self.tape = tape
        self.cond = cond

    def __enter__(self):
        if self.tape is not None and self.cond:
            self.tape.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tape is not None and self.cond:
            self.tape.__exit__(exc_type, exc_value, traceback)
