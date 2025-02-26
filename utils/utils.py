
def still_gaining(train_loss):
    print("still_gaining:", train_loss)
    if len(train_loss) <= 1 or train_loss[-1] < train_loss[-2]:
        return True

    return False

