##class CallbackList(object):
##    """ This actually runs the callbacks specified by user """
##    def __init__(self, callbacks=[], queue_length=10):
##        self.callbacks = [c for c in callbacks]
##        self.queue_length = queue_length
##
##    def append(self, callback):
##        self.callbacks.append(callback)
##
##    def _set_params(self, params):
##        for callback in self.callbacks:
##            callback._set_params(params)
##
##    def _set_model(self, model):
##        for callback in self.callbacks:
##            callback._set_model(model)
##
##    def on_epoch_begin(self, epoch, logs={}):
##        for callback in self.callbacks:
##            callback.on_epoch_begin(epoch, logs)
##        self._delta_t_batch = 0.
##        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
##        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)
##
##    def on_epoch_end(self, epoch, logs={}):
##        for callback in self.callbacks:
##            callback.on_epoch_end(epoch, logs)
##
##    def on_batch_begin(self, batch, logs={}):
##        t_before_callbacks = time.time()
##        for callback in self.callbacks:
##            callback.on_batch_begin(batch, logs)
##        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
##        delta_t_median = np.median(self._delta_ts_batch_begin)
##        if self._delta_t_batch > 0. and delta_t_median > 0.95 * \
##           self._delta_t_batch and delta_t_median > 0.1:
##            warnings.warn('Method on_batch_begin() is slow compared '
##                          'to the batch update (%f). Check your callbacks.'
##                          % delta_t_median)
##        self._t_enter_batch = time.time()
##
##    def on_batch_end(self, batch, logs={}):
##        if not hasattr(self, '_t_enter_batch'):
##            self._t_enter_batch = time.time()
##        self._delta_t_batch = time.time() - self._t_enter_batch
##        t_before_callbacks = time.time()
##        for callback in self.callbacks:
##            callback.on_batch_end(batch, logs)
##        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
##        delta_t_median = np.median(self._delta_ts_batch_end)
##        if self._delta_t_batch > 0. and (delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1):
##            warnings.warn('Method on_batch_end() is slow compared '
##                          'to the batch update (%f). Check your callbacks.'
##                          % delta_t_median)
##
##    def on_train_begin(self, logs={}):
##        for callback in self.callbacks:
##            callback.on_train_begin(logs)
##
##    def on_train_end(self, logs={}):
##        for callback in self.callbacks:
##            callback.on_train_end(logs)


class Callback(object):
    '''Abstract base class used to build new callbacks.
    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.
    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.
    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:
        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    '''
    def __init__(self):
        pass

    def _set_params(self, params):
        self.params = params

    def _set_model(self, model):
        self.model = model

    def on_episode_begin(self, episode, logs={}):
        pass

    def on_episode_end(self, episode, logs={}):
        pass
    
    def on_allepisodes_begin(self, logs={}):
        pass

    def on_allepisodes_end(self, logs={}):
        pass

    def on_act_begin(self, act, logs={}):
        pass

    def on_act_end(self, act, logs={}):
        pass

class BaseCallback(Callback):
    def on_episode_begin(self, episode, logs={}):
         self.params['observation'] = self.params['env'].reset()

    def on_act_end(self, act, logs):
        self.params['observation'] = logs['observation']
        if logs['done']:
            
         

class EnvRender(Callback):
    def on_act_begin(self, act, logs={}):
        self.params['env'].render()

    def on_allepisodes_end(self, logs={}):
        self.params['env'].render(close=True)


class EnvMonitor(Callback):
    def on_allepisodes_begin(self, logs={}):
        try:
            self.params['env'].monitor.start(args.env_monitor_dir, force=args.env_monitor_dir_overwrite)
        except gym.error.Error as ge:
            sys.exit("\nError: Won't overwrite monitor logs at {0}. Use '--env_monitor_dir_overwrite=True' to force overwrite.".format(args.env_monitor_dir, ))

    def on_allepisodes_end(self, logs={}):
        self.params['env'].monitor.close()


class Statistics(Callback):
    def on_act_end(self, epoch, logs={}):
        

#class StandardiseData(Callback):
    #def on_act_begin(self, logs={}):
    #standardise_input_data
