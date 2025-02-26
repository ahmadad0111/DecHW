from objprint import add_objprint


@add_objprint
class Message:
    def __init__(self, time, source, model=None, outs=None, data_size=None, gradients=None, hessian_diag=None):
        self._time = time
        self._source = source
        # optional fields
        self._model = model
        self._outs = outs
        self._dataset_size= data_size
        self._gradients = gradients
        self._hessian_diag = hessian_diag
        

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        self._time = time

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        self._source = source

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def outs(self):
        return self._outs
    
    @outs.setter
    def outs(self,probs,counts):
        self._outs = (probs,counts)
    
    @property
    def dataset_size(self):
        return self._dataset_size
    
    @dataset_size.setter
    def dataset_size(self,data_count):
        self._dataset_size = data_count
    @property
    def gradients(self):
        return self._gradients

    @gradients.setter
    def gradients(self, grads):
        self._gradients = grads

    @property
    def hessian_diag(self):
        return self._hessian_diag
    @hessian_diag.setter
    def hessian_diag(self, hessian_diag):
        self._hessian_diag = hessian_diag


   

    

    #

    def __lt__(self, other):
        # definition of the < operator:  sort first by time of event, then by node_id
        return (self.time < other.time)

    def __iter__(self):
        ''' Returns the Iterator object '''
        return iter([self.time, self.source, self.model])


# class MessageBuffer:
#     def __init__(self):
#         self._msg_buffer = deque()
#
#     def
