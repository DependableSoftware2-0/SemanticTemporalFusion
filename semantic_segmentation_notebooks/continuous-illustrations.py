#
# Example file for the EfProb Manual
# with continous classical probability examples.
#
# Copyright Bart Jacobs, Kenta Cho
# Radboud University Nijmegen
# efprob.cs.ru.nl
#
# Date: 2018-04-24
#
from efprob_dc import *
from math import *

def states():

    print("\nSection: States\n")

    print("* print uniform state")
    u = uniform_state(R(0,5))
    print( u )
    #u.plot()

    print("\n===\n")

    print("* Validity of truth predicate")
    s = State(lambda x: 0.5 * x, R(0,2))
    print( s >= truth(R(0,2)) )

    print("\n===\n")

    print("* Infinite domain")
    u = State(lambda x: e ** -x, R(0, inf))
    print( u >= truth(u.dom) )
    v = State(lambda x: e ** x, R(-inf, 0))
    print( v >= truth(v.dom) )
    w = State(lambda x: 1/sqrt(pi) * e ** (-x*x), R)
    print( w >= truth(w.dom) )

    print("\n===\n")

    print("* Product domains")
    s = State(lambda x,y: 4 * x * y, [R(0,1), R(0,1)])
    print( s >= truth(s.dom) )
    t = State(lambda x,y,z: 8/15 * x * y * z, [R(0,1), R(1,2), R(2,3)])
    print( t >= truth(t.dom) )

    print("\n===\n")

    print("* predefined distributions")
    s = gaussian_state(2, 1)
    print( s )
    t = gaussian_state(2, 1, R(-10,10))
    print( t >= truth(t.dom) )
    #t.plot()
    be = beta_state(2,5)
    #be.plot()
    ga = gamma_state(5,2)
    #ga.plot(R(0,10))
    ex = exponential_state(2)
    #ex.plot(R(0,5))

def operations_on_states():

    print("\nSubsection: Operations on states\n")

    print("* Product of states")
    s = uniform_state(R(-1,1)) @ uniform_state(R(5,10))
    print( s )
    t = flip(0.3) @ s
    print( t )

    w = gaussian_state(1, 0.4, R(0,2)) @ beta_state(2,1)
    #w.plot()


def predicates():

    print("\nSection: Predicates\n")

    print("* Predicate definition")
    q = Predicate(lambda x: 1 if x < 0 else 0.5, R(-1,1))
    u = uniform_state(R(-1,1))
    print( u >= q )
    s = u @ flip(0.4)
    print( s >= ~q @ yes_pred )

    print("\n===\n")

    print("* Fires on a ship")    
    fires = exponential_state(1) @ exponential_state(1)
    saved = Predicate(lambda t1, t2: 1 if t1 < 1.25 and t1 + 0.25 * t2 < 1.375 
                      else 0, [R(0,inf), R(0,inf)])
    print( fires >= saved )
    print("Alternative representation, using logic in EfProb")
    p1 = Predicate(lambda t: t < 1.25, R(0,inf))
    p2 = Predicate(lambda t1, t2: 1 if t1 + 0.25 * t2 < 1.375 else 0, 
                   [R(0,inf), R(0,inf)])
    #print( fires >= (p1 @ truth(R(0,inf))) & p2 )
    


def conditioning():

    print("\nSubsection: Conditioning\n")

    print("* Archeological tomb")
    dom = R(0,100)
    p1 = Predicate(lambda x: 0.6 * e ** (-1/2000 * (x - 20) ** 2), dom)
    p2 = Predicate(lambda x: 0.9 * e ** (-1/1500 * (x - 50) ** 2), dom)
    p3 = Predicate(lambda x: 0.8 * e ** (-1/1000 * (x - 90) ** 2), dom)
    u = uniform_state(dom)
    #u.plot()
    findings = [1, 3, 2, 1, 1]
    def pred(f): return p1 if f==1 else p2 if f==2 else p3
    for f in findings: u = u / pred(f)
        #u.plot()
    print("Expected age: ", u.expectation() )


def random_variables():

    print("\nSection: Random Variables\n")

    print("* Random variable validity")
    s = uniform_state(R(0,5))
    rv = RandVar(lambda x: x, R(0,5))
    print( s >= rv )

    print("\n===\n")

    print("* Gaussian expectation and standard deviation")
    g = gaussian_state(-1.4, 2.9)
    print( g.expectation() )
    print( g.st_deviation() )

    print("\n===\n")

    print("* Expectation and variance")
    t = State(lambda x: 10/3 * x - 10/3 * x ** 4, R(0,1))
    print( t >= truth(R(0,1)) )
    print( t.expectation(), 5/9 )
    print( t.variance(), 55/1134 )

    print("\n===\n")

    print("* Covariance of a product state")
    w = State(lambda x,y: 4*x*y, [R(0,1), R(0,1)])
    print( w >= truth(R(0,1)) @ truth(R(0,1)) )
    print( w.covariance() )
    print( w.correlation() )


def state_pred_transformation():

    print("\nSubsection: State and predicate transformation\n")

    print("\n===\n")

    print("* Coin parameter learning")
    bias_dom = R(0,1)
    prior = uniform_state(bias_dom)
    chan = chan_fromklmap(lambda r: flip(r), bias_dom, bool_dom)
    print( chan >> prior )
    observations = [0,1,1,1,0,0,1,1]
    s = prior
    # s.plot()
    for ob in observations:
        pred = yes_pred if ob==1 else no_pred
        s = s / (chan << pred)
        # s.plot()
    print( chan >> s )
    print( s.expectation() )

    print("\n===\n")

    print("* Advertisement analysis")
    N = 50
    c = chan_fromklmap(lambda r: binomial(N, r), R(0,1), range(N+1))
    prior = uniform_state(R(0,1))
    scoreA = 18
    scoreB = 27
    postA = prior / (c << point_pred(scoreA, range(N+1)))
    postB = prior / (c << point_pred(scoreB, range(N+1)))
    #postA.plot()
    #postB.plot()
    profitA = randvar_fromfun(lambda x: x * 100 - N * 10, range(N+1))
    profitB = randvar_fromfun(lambda x: x * 100 - N * 30, range(N+1))
    print("Profit for campaign A: ", (c >> postA).expectation(profitA))
    print("Profit for campaign B: ", (c >> postB).expectation(profitB))
    print( postA >= (c << profitA) )
    print( postB >= (c << profitB) )
    print( scoreA * 100 - N * 10 )
    print( scoreB * 100 - N * 30 )


def all():
    states()
    operations_on_states()
    predicates()
    conditioning()
    random_variables()
    state_pred_transformation()



def main():
    all()
    #states()
    #operations_on_states()
    #predicates()
    #conditioning()
    #random_variables()
    #state_pred_transformation()


if __name__ == "__main__":
    main()

