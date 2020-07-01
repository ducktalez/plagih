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
    message_pretxt = f'{BColors.FAIL}ERROR: {BColors.WARNING}'
    print(f'{message_pretxt}{BColors.MAGENTA}{text}{BColors.RESET}')


def print_warning(message_type, text, print_type=None, time_total=0.0):
    """
    Printing warnings
    """
    printez(message_type, text, print_type=print_type, time_total=time_total)


def print_blue(*args):
    print('{}{}{}'.format(BColors.CYAN, ''.join(args), BColors.RESET))
    return


def printez(message_type, text, print_type=None, time_total=0.0):
    """
    giving prints colours, accessable from everywhere
    """
    if print_type:
        if message_type not in print_type:
            return

    message_pretxt = BColors.RESET
    message_posttxt = BColors.RESET
    if 'i' in message_type:
        message_style = BColors.CYAN
        message_pretxt = ''
    elif 'e' in message_type:
        message_style = BColors.RED
        message_pretxt = 'ERROR: '
    elif 'w' in message_type:
        message_style = BColors.WARNING
        message_pretxt = ''
    elif 'g' in message_type:
        message_style = BColors.BLUE
        message_pretxt = f'{time_total:3.0f}s. '  # sfeh current time instead and local time at the end?
    elif 'v' in message_type:
        message_style = BColors.WHITE
        message_pretxt = 'Verbose: '
    elif 'f' in message_type:
        message_style = BColors.MAGENTA
        message_pretxt = 'Writing File: '
    elif 'a' in message_type:
        message_style = BColors.GREEN
        message_pretxt = ''
    else:
        message_style = ''
        printez('w', f'print_type-mode {message_type} not known.')

    print(f'{message_style}{message_pretxt}{text}{message_posttxt}')
    return

