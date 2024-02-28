import sys
import time


class Progress():
    ''' Shows the progress of the iterative process as a porgressbar

    Attributes
    ----------
    start (float): Starting time for computing the eta
    bars    (int): Number of bars in the progressbar
    iterations (int): Total number of iterations
    verbose (bool): If true writes more details of the progress on the screen.
    '''

    def __init__(self, iterations, verbose):
        self.start = time.time()
        self.bars  = 50
        self.iterations = iterations
        self.verbose = verbose

    def __call__(self, j=0, loss=0.0, total_time=False):
        if not total_time:
            percent = int(round(j / self.iterations, 2) * 100)
            if self.verbose:
                print(f'------ Epoch {j:^5d}/{self.iterations:^5d} '+
                      f'({percent:2d}%): Loss = {loss:^12.9f}')
            else:
                eta = 0.0 if j == 0 else ((time.time() - self.start) / j) * (self.iterations - j)
                mins, sec = divmod(eta, 60)
                hrs, mins = divmod(mins, 60)
                time_str  = f'{int(hrs):02d}:{int(mins):02d}:{int(sec):02d}'
                bar_j = int(self.bars * j / self.iterations)
                print(f'[{u"█"*bar_j}{("."*(self.bars - bar_j))}] {j}/{self.iterations}'+
                      f'({percent:2d}%)[ETA {time_str}]: Loss = {loss:^7.6f}',
                      end='\r', file=sys.stdout, flush=True)
        else:
            eta = time.time() - self.start
            mins, sec = divmod(eta, 60)
            hrs, mins = divmod(mins, 60)
            time_str  = f'{int(hrs):02d}:{int(mins):02d}:{int(sec):02d}'
            print(f'\n------ Total time: {time_str}')
            time_step = eta / self.iterations
            print(f'------ Time/step: {time_step:^6.1} s')

