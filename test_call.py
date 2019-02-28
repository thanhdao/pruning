# test_call.py
class TestClass:
  def train(self):
    print('train')

  def prune(self):
    print('prune')

input = 'train'
test = TestClass()
method = getattr(test, input)
method()