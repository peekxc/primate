import primate

def test_include():
  include_path = primate.get_include()[-15:]
  assert 'primate' in include_path
  assert 'include' in include_path
  
  