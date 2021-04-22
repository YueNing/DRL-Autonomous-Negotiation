from drl_negotiation.utils import show

def test_show():
    filename = "/tmp/policy2/_rewards.pkl"
    show(filename)

if __name__ == "__main__":
    test_show()