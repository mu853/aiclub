confirm loss:
  python main.py loss <data.csv> <bgimg.png> [output.png]
  python t1.py <data.csv> <bgimg.png> [output.png]

classify with AutoEncoder (k=2-9):
  python main.py pred <data.csv> <bgimg.png> [output.png]
  python t2.py <data.csv> <bgimg.png> [output.png]

confirm accuracy:
  python main.py acc <data.csv> <bgimg.png>
  python t3.py <data.csv> [output.png]

classify with k-mean (k=2-9):
  python t4.py <data.csv> <bgimg.png>

