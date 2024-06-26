import sqlite3
import os


if os.path.exists('reviews.sqlite'):
    os.remove('reviews.sqlite')

conn = sqlite3.connect('reviews.sqlite')

c = conn.cursor()
c.execute('create table review_db' ' (review TEXT, sentiment INTEGER, date TEXT)')

example1 = 'I love this movie'

c.execute("INSERT INTO review_db" " (review, sentiment, date) VALUES" " (?, ?, DATETIME('now'))", (example1, 1))

example2 = 'I disliked this movie'

c.execute("INSERT INTO review_db" " (review, sentiment, date) VALUES" " (?, ?, DATETIME('now'))", (example2, 0))

conn.commit()

conn.close()