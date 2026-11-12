from bootstrap import ensure
if __name__=='__main__':
    ensure('cpu','CPU.py')
    from client import run
    try:run('cpu')
    except KeyboardInterrupt:print('\nStopped. Reserved lesson uses are not recycled.')
