# no imports here


class BColors:  # sfeh can be deleted
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[39m'

    BLACK2 = '\033[40m'
    RED2 = '\033[41m'


def print_e(text):
    """
    Printing errors
    """
    print(f'{BColors.RED}ERROR: {BColors.FAIL}{text}{BColors.RESET}')


def print_warning(message_type, text, print_type=None, time_total=0.0):
    """
    Printing warnings
    """

    if print_type:
        if message_type not in print_type:
            return
    if message_type == 'w':
        print(f'{BColors.WARNING}Warning ({message_type}): {text}{BColors.RESET}')
    else:
        print(f'{BColors.WARNING}Warning ({message_type}):{BColors.RESET} {text}')
    return


def print_blue(txt):
    print(f"{BColors.CYAN}{txt}{BColors.RESET}")
    return


def printez(message_type, text, print_type=None, time_total=0.0):
    """
    giving prints colours, accessable from everywhere
    """
    if print_type:
        if message_type not in print_type:
            return

    if 'i' in message_type:
        pre_msg = BColors.CYAN
    elif 'g' in message_type:
        pre_msg = f'{time_total:3.0f}s. '  # sfeh current time instead and local time at the end?
        # pre_msg = BColors.WHITE
    elif 'f' in message_type:
        pre_msg = f'{BColors.MAGENTA}Writing File: '
    elif 'a' in message_type:
        pre_msg = f'{BColors.GREEN}Paretofront: '
    else:
        raise Exception(f'print_type-mode {message_type} not known.')

    print(f'{pre_msg}{text}{BColors.RESET}')
    return

