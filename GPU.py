from bootstrap import ensure
if __name__=='__main__':
    ensure('gpu','GPU.py')
    from client import run
    try:run('cuda')
    except KeyboardInterrupt:print('\nStopped. Reserved lesson uses are not recycled.')
