# JURI

Install requirements using `pip install -r requirements.txt`. We recommend using `virtualenv` or `pyenv`.

Before you start, you should create `config/__init__.py` and `config/config.py`. Edit `config/config.py` as shown below:
```python
class Config:
    # MySQLdb doesn't support Python 3
    SQLALCHEMY_DATABASE_URI = 'your mysql address'  # change that
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    TESTING = True
    DEBUG = True
    SECRET_KEY = 'your secret key'  # change that
    ]
```

Then you can run `python manage.py create` to create databases.

To download latest data from HUDOC, run:
```bash
python -m crawler.hudoc COMMUNICATEDCASES -u
python -m crawler.hudoc DECISIONS -u
python -m crawler.hudoc JUDGMENTS -u
python -m crawler.hudoc JUDGMENTS -d
```
Now that the data is ready, you can apply models and make predictions. If you want to use the random guessing model for now, run:
```bash
python -m model.run
```

You can use `python manage.py runserver` and check if everything is working correctly in browser now.
