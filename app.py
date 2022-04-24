from flask import *

from src.DataSource import *
from src.Utils import *
from src.Algorithms import *


App = Flask(__name__)
data_source = WHappinessDataSrc('./datasets/happiness-2019.csv')


def pritify(text: str):
    cap = ''.join(x if x[-1].isupper() else x[0].upper() for x in text.split("_"))
    return cap if len(cap) > 2 else remove_under(text)


def remove_under(text: str):
    return ' '.join(text.split("_"))


App.jinja_env.globals.update(pritify=pritify)
App.jinja_env.globals.update(remove_under=remove_under)


def main():
    App.run()


@App.get('/')
def main_page():
    req = RequestParser.parse(request)

    if req.method == SEQUENTIAL:
        data, seq_acc, rand_acc, time_elapsed = top_k_sequential(data_source, req)
    else:
        data, seq_acc, rand_acc, time_elapsed = top_k_threshold(data_source, req)

    return render_template('index.html', val_cols=VAL_COLS, columns=COLS, time_elapsed=time_elapsed,
                           total_acc=seq_acc+rand_acc, seq_acc=seq_acc, data=data,  k_value=req.k_value)


if __name__ == '__main__':
    main()
