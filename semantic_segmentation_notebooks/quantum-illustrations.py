#
# Example file for the EfProb Manual
# with quantum classical probability examples.
#
# Copyright Bart Jacobs, Kenta Cho
# Radboud University Nijmegen
# efprob.cs.ru.nl
#
# Date: 2018-04-24
#
from efprob_qu import *
from math import *

def states():
    print("\nSection: States\n")
    print("* printing basic states")
    print( ket(0) )
    print( ket(1) )
    print( plus )
    print( minus )

    print("\n===\n")

    print("* Bloch state equalities")
    print( ket(0) == bloch_state(0,0) )
    print( ket(1) == bloch_state(pi,0) )
    print( plus == bloch_state(pi/2, 0) )
    print( minus == bloch_state(pi/2, pi) )




def operations_on_states():
    print("\nSubsection: Operations on states\n")

    print("* Printing product states")
    print( ket(0) @ ket(1) )
    
    print( ket(0) @ plus )

    print("\n===\n")

    print("* printing product states")
    print( ket(0,1,1) )

    print("\n===\n")

    print("* Printing product states")
    print( ket(0) ** 3 )

    print("\n===\n")

    print("* Printing product states")
    print( (plus @ ket(0)) ** 4 )

    print("\n===\n")

    print("* Printing convex sum example")
    print( convex_state_sum( (0.2,ket(0)), (0.3,plus), (0.5,minus) ) )
    print( convex_state_sum( (1/2, plus), (1/2, minus) ) )

    print("\n===\n")

    print("\n===\n")

    print("* Marginalisation")
    print( ket(0) @ ket(1) @ plus @ minus % [1,0,1,0] )

    print("\n===\n")



def basic_states():
    print("\nSubsection: Basic states\n")

    print("* Probabilistic state, normalised")
    print( probabilistic_state(3,4,1) )

    print("\n===\n")

    print("* Classical flip")
    print( cflip(0.3) )

    print("\n===\n")

    print("* Product of these")
    print( cfflip @ probabilistic_state(2,2,1) )

    print("\n===\n")

    print("* Vector state")
    print( vector_state(1, complex(1, 2), -2) )

    print("\n===\n")

    print( vector_state(0.5 * sqrt(3), complex(0, 0.5)) )

    print("\n===\n")

    print("* Point/unit state")
    print( point_state(2,3) )

    print("\n===\n")

    print("* Random state")
    print( random_state([3]) )

    print("\n===\n")

    print("* Random probabilistic state")
    print( random_probabilistic_state([2]) )




def predicates():

    print("\nSection: Predicates\n")

    print("* Falsity and truth")
    print( falsity([2]) )
    print( truth([2,3]) )

    print("\n===\n")

    print("* Probabilistic predicate, of length 4")
    print( probabilistic_pred(0.1,1,0.25,0.0) )

    print("\n===\n")

    print("* Unit predicate, of length 4")
    print( point_pred(2,4) )

    print("\n===\n")



def operations_on_predicates():

    print("\nSubsection: Operations on predicates\n")

    print("* Random predicate and its orthocomplement")
    p = random_pred([2])
    print(p)
    print(~p)

    print("\n===\n")

    print("* Scaled probabilistic predicate")
    print( 0.3 * probabilistic_pred(0.2,0.5) )

    print("\n===\n")

    print("* Sum of unit predicates")
    print( point_pred(2,4) + point_pred(0,4) )
    print( x_pred )
    p = vector_pred(1/sqrt(2), 1/sqrt(2))
    print( p )
    print( x_pred + p )

    print("\n===\n")

    print("* Bell states form a test")
    print( bell00.as_pred() + bell01.as_pred() + 
           bell10.as_pred() + bell11.as_pred() )

    print("\n===\n")

    print("* Also GHZ states form a test")
    print( ghz1.as_pred() + ghz2.as_pred() + ghz3.as_pred() + ghz4.as_pred() +
           ghz5.as_pred() + ghz6.as_pred() + ghz7.as_pred() + ghz8.as_pred() )


    print("\n===\n")

    print("* Parallel conjunction of truth and falsity")
    print( truth([2]) @ falsity([3]) )

    print("\n===\n")

    print("* Sequential conjunction of two probabilistic predicates")
    print( probabilistic_pred(0.2,0.8) & probabilistic_pred(0.4, 0.6) )

    print("\n===\n")

    print("* Weakening illustration")
    s = cflip(0.4)
    t = ket(0,0)
    p = point_pred(1,2)
    print( s >= p )
    print( s @ t >= p @ truth(t.dom) )



def validity():

    print("\nSubsection: Validity\n")

    print("* Validity example")
    v = vector_pred(0.5 * sqrt(3), complex(0, 0.5)) 
    print( v )
    print( ket(0) >= v )
    print( ket(1) >= v )

    print("\n===\n")

    print("* Validity preserves scaling example")
    # s = random_state([100])
    # p = random_pred([100])
    # print( s >= 0.3 * p )
    # print( 0.3 * (s >= p) )

    print("\n===\n")

    print("* Commutativity of sequential conjunction")
    p = point_pred(0,2)
    q = plus.as_pred()
    print( ket(0) >= p & q )
    print( ket(0) >= q & p )

    print("\n===\n")

    print("* Validity of states")
    s = random_state([8])
    t = random_state([8])
    print( s >= t.as_pred() )
    print( t >= s.as_pred() )

    print("\n===\n")

    print("* Validity of vector states")
    v1 = np.random.rand(5)
    v2 = v1/np.linalg.norm(v1)
    w1 = np.random.rand(5)
    w2 = w1/np.linalg.norm(w1)
    s = vector_state(*v2)
    t = vector_state(*w2)
    print( s >= t.as_pred() )
    print( np.inner(v2, w2) ** 2 )

    print("\n===\n")

    print("* Bell table")
    v1 = vector_pred(1/sqrt(2), 1/sqrt(2))
    A1 = v1
    B1 = v1
    v2 = vector_pred(1/sqrt(2), 0.5/sqrt(2) * complex(1, -sqrt(3)))
    A2 = v2
    B2 = v2
    print( A1 )
    print( A2 )
    print("")
    print( bell00 >= A1 @ B1, bell00 >= A1 @ ~B1, bell00 >= ~A1 @ B1, bell00 >= ~A1 @ ~B1 )
    print("")
    print( bell00 >= A1 @ B2, bell00 >= A1 @ ~B2, bell00 >= ~A1 @ B2, bell00 >= ~A1 @ ~B2 )
    print("")
    print( bell00 >= A2 @ B1, bell00 >= A2 @ ~B1, bell00 >= ~A2 @ B1, bell00 >= ~A2 @ ~B1 )
    print("")
    print( bell00 >= A2 @ B2, bell00 >= A2 @ ~B2, bell00 >= ~A2 @ B2, bell00 >= ~A2 @ ~B2 )

    print("\nAdditional experiments")
    print("Validities of B1, in itself and after conditiong with A1, A2")
    print( bell00 >= truth([2]) @ B1, 
           bell00 / (A1 @ truth([2])) >= truth([2]) @ B1,
           bell00 / (A2 @ truth([2])) >= truth([2]) @ B1 )
    print("Validities of B2, in itself and after conditiong with A1, A2")
    print( bell00 >= truth([2]) @ B2, 
           bell00 / (A1 @ truth([2])) >= truth([2]) @ B2,
           bell00 / (A2 @ truth([2])) >= truth([2]) @ B2 )
    print("Validities of A1, in itself and after conditiong with B1, B2")
    print( bell00 >= A1 @ truth([2]), 
           bell00 / (truth([2]) @ B1) >= A1 @ truth([2]),
           bell00 / (truth([2]) @ B2) >= A1 @ truth([2]) )
    print("Validities of A2, in itself and after conditiong with B1, B2")
    print( bell00 >= A2 @ truth([2]), 
           bell00 / (truth([2]) @ B1) >= A2 @ truth([2]),
           bell00 / (truth([2]) @ B2) >= A2 @ truth([2]) )

    print("\nphi's")
    phi1 = A1 @ B1 + ~A1 @ ~B1
    phi2 = A1 @ B2 + ~A1 @ ~B2
    phi3 = A2 @ B1 + ~A2 @ ~B1
    phi4 = A2 @ ~B2 + ~A2 @ B2

    print( bell00 >= phi4 & phi1 & phi3 & phi2 )

    print( (bell00 >= phi1) + (bell00 >= phi2) + (bell00 >= phi3) + (bell00 >= phi4) )





def conditioning():

    print("\nSubsection: Conditioning\n")

    print("* Bayes' rule illustration")
    s = random_state([2])
    p = random_pred([2])
    q = random_pred([2])
    print( s / p >= q )
    print( (s >= p & q) / (s >= p) )

    print("\n===\n")

    print("* polarisation illustration, in several steps")
    s = random_state([2])
    vert_filt = point_pred(0,2)
    hor_filt = point_pred(1,2)
    diag_filt = plus.as_pred()
    print( s >= vert_filt )
    s_vert = s / vert_filt

    print("\n===\n")

    print( s_vert >= vert_filt )
    print( s_vert >= hor_filt )

    print("\n===\n")

    print( s_vert >= diag_filt & hor_filt )

    print("\n===\n")

    print( s_vert >= hor_filt & diag_filt )

    print("\n===\n")

    print( s_vert / diag_filt >= hor_filt )

    print("\n===\n")

    print("* Iterated conditioning is not the same as conditioning by conjunction")
    print( s / vert_filt / diag_filt )
    print( s / (vert_filt & diag_filt) )

    print("\n===\n")

    print("* Failure of the law of total probability")

    print( s >= hor_filt )
    print( (s / diag_filt >= hor_filt) * (s >= diag_filt) + 
           (s / ~diag_filt >= hor_filt) * (s >= ~diag_filt) )
    print( (s >= diag_filt & hor_filt) + (s >= ~diag_filt & hor_filt) )

    print("\n===\n")

    print("* Failure of associativity of sequential conjunction")
    print( s / vert_filt >= vert_filt & (diag_filt & hor_filt) )
    print( s / vert_filt >= (vert_filt & diag_filt) & hor_filt )

    print("\n===\n")

    print("* EPR Crossover")
    print( bell00 >= truth([2]) @ point_pred(0, 2) )
    print( bell00 / (point_pred(0, 2) @ truth([2])) >= truth([2]) @ point_pred(0, 2) )
    print( bell00 / (point_pred(1, 2) @ truth([2])) >= truth([2]) @ point_pred(0, 2) )
    print("\n===\n")

    print("* Crossover via the x predicates")
    x_pp = vector_pred(-1/sqrt(2), 1/sqrt(2))
    x_mp = vector_pred(1/sqrt(2), 1/sqrt(2))
    print( bell00 >= truth([2]) @ x_pp )
    print( bell00 / (x_pp @ truth([2])) >= truth([2]) @ x_pp )
    print( bell00 / (x_mp @ truth([2])) >= truth([2]) @ x_pp )
    print("Similar outcomes for the other Bell states")
    print( "bell01 ", 
           bell01 >= truth([2]) @ x_pp,
           bell01 / (x_pp @ truth([2])) >= truth([2]) @ x_pp,
           bell01 / (x_mp @ truth([2])) >= truth([2]) @ x_pp )
    print( "bell10 ", 
           bell10 >= truth([2]) @ x_pp,
           bell10 / (x_pp @ truth([2])) >= truth([2]) @ x_pp,
           bell10 / (x_mp @ truth([2])) >= truth([2]) @ x_pp )
    print( "bell11 ", 
           bell11 >= truth([2]) @ x_pp,
           bell11 / (x_pp @ truth([2])) >= truth([2]) @ x_pp,
           bell11 / (x_mp @ truth([2])) >= truth([2]) @ x_pp )

    print("\n===\n")

    print("* Crossover conditioning")
    print( (bell00 / (point_pred(0,2) @ truth([2]))) % [0,1] == point_state(0,2),
           (bell00 / (point_pred(1,2) @ truth([2]))) % [0,1] == point_state(1,2) )
    print( (bell00 / (x_pp @ truth([2]))) % [0,1] == x_plus,
           (bell00 / (x_mp @ truth([2]))) % [0,1] == x_min )


def random_variables():

    print("\nSubsection: Operations on random variables\n")

    print("* Types of scalar multiplications")
    print( type( ket(0) ) )
    print( type( 0.5 * ket(0).as_pred() ) )
    print( type( 5 * ket(0).as_pred() ) )

    print("\n===\n")

    print("* Types of scalar additions")
    print( type( point_pred(0,2) ) )
    print( type( point_pred(0,2) + point_pred(0,2) ) )

    print("\n===\n")

    print("* Dice variance, quantum style")
    dice = uniform_probabilistic_state([6])
    points = 1 * point_pred(0,6) + 2 * point_pred(1,6) + 3 * point_pred(2,6) \
             + 4 * point_pred(3,6) + 5 * point_pred(4,6) + 6 * point_pred(5,6)
    print( dice >= points )
    print( dice.variance(points) )

    print("\n===\n")

    print("* Bell inequality")
    x_rv = ~x_pred - x_pred 
    y_rv = ~y_pred - y_pred 
    z_rv = z_pred - ~z_pred 
    # Alternative, with same outcome:
    # x_rv = RandVar(x_matrix, [2])
    # y_rv = RandVar(y_matrix, [2])
    # z_rv = RandVar(z_matrix, [2])
    Q = z_rv
    R = x_rv
    S = 1/math.sqrt(2) * (-z_rv - x_rv)
    T = 1/math.sqrt(2) * (z_rv - x_rv)
    print("\nEigenvalues of random variables")
    print(np.linalg.eigh(Q.array)[0])
    print(np.linalg.eigh(R.array)[0])
    print(np.linalg.eigh(S.array)[0])
    print(np.linalg.eigh(T.array)[0])
    psi = bell11
    print( psi >= Q @ S, psi >= R @ S, psi >= R @ T, psi >= Q @ T )
    print("Bell violation: ", 2 < (psi >= Q @ S + R @ S + R @ T - Q @ T) )
    A1 = -x_rv
    A2 = -y_rv
    B1 = 1/math.sqrt(2) * (x_rv + y_rv)
    B2 = 1/math.sqrt(2) * (x_rv - y_rv)
    print("Violations of monotonicity of @:", 
          is_positive(truth([2]).array - Q.array),
          (psi >= Q @ S) > (psi >= truth([2]) @ S) )
    print("Alternative Bell violation: ",
          psi >= A1 @ B1 + A1 @ B2 + A2 @ B1 - A2 @ B2 )


def state_transformation():

    print("\nSubsection: State transformation\n")

    print("* plus and minus states via state transformation")
    plus = hadamard >> ket(0)
    minus = hadamard >> ket(1)
    print( plus )
    print( minus )

    print("\n===\n")

    print("* cnot channel applied to various ket combinations")
    print( cnot >> ket(0,0) )
    print( cnot >> ket(0,1) )
    print( cnot >> ket(1,0) )
    print( cnot >> ket(1,1) )

    print("\n===\n")

    print("* x/y/z channels on Bloch states")
    t = random.uniform(0,pi)
    p = random.uniform(0,2*pi)
    print( x_chan >> bloch_state(t, p) == bloch_state(pi - t, 2*pi - p) )
    print( y_chan >> bloch_state(t, p) == bloch_state(pi - t, pi - p) )
    print( z_chan >> bloch_state(t, p) == bloch_state(t, pi + p) )


def predicate_transformation():

    print("\nSubsection: Predicate transformation\n")

    print("* Predicates used in Bell table, via predicate transformation")
    A1 = hadamard << point_pred(0,2)
    print( A1 )
    angle = pi / 3
    A2 = phase_shift(angle) << A1
    print( A2 )

    print("\n===\n")

    print("* Transformations validity")
    p = random_pred([2])
    s = random_state([2])
    print( x_chan >> (hadamard >> s) >= p )
    print( (x_chan * hadamard) >> s >= p )
    print( s >= hadamard << (x_chan << p) )
    print( s >= (x_chan * hadamard) << p )

    print("\n===\n")

    print("* Classical discrete and quantum channels")
    # consider discrete channel c : 2 -> 3 given by
    # c(0) = 1/2|0> + 1/8|1> + 3/8|2>,  c(1) = 1/3|0> + 1/2|1> + 1/6|2>
    # for state s = 1/5|0> + 4/5|1> we get 
    # c >> s = 11/30|0> + 17/40|1> + 5/24|2> where
    # 11/30 = 0.366.., 17/40 = 0.425, 5/24 = 0.20833..
    mat = np.array([ [ np.array([[1/2,0], [0,1/3]]), 
                       np.array([[0,0], [0,0]]), 
                       np.array([[0,0], [0,0]]) ],
                     [ np.array([[0,0], [0,0]]), 
                       np.array([[1/8,0], [0,1/2]]), 
                       np.array([[0,0], [0,0]]) ],
                     [ np.array([[0,0], [0,0]]), 
                       np.array([[0,0], [0,0]]),
                       np.array([[3/8,0], [0,1/6]]) ] ])
    d = Channel(mat, [2], [3])
    print( d << truth([3]) )
    print( d >> probabilistic_state(1/5, 4/5) )
    print( d << probabilistic_pred(1/2, 0, 1) )


def structural_channels():
    
    print("\nSubsection: Structural channels\n")

    print("* Plus state via hadamard channel")
    c = (hadamard @ idn([2])) * cnot
    print( c >> ket(0,0) )

    print("\n===\n")

    print("* Outcome of discarding")
    print( discard([2]) >> ket(0) )
    print( discard([2]) >> ket(1) )

    print("\n===\n")

    print("* Projection, in various forms")
    s = random_state([2])
    print( s )
    t = random_state([2])
    #print( (proj1 * cnot) >> (s @ t) )
    print( ((idn([2]) @ discard([2])) * cnot) >> (s @ t) )
    print( (cnot >> (s @ t)) % [1,0] )
    print( ((idn([2]) @ discard([2])) * cnot) >> (s @ t) == (cnot >> (s @ t)) % [1,0] )

    print("\n===\n")

    print("* Swap examples")
    s = random_state([2])
    t = random_state([2])
    print( swap >> (s @ t) == t @ s )
    print( swap >> bell00 == bell00,
           swap >> bell01 == bell01,
           swap >> bell10 == bell10,
           swap >> bell11 == bell11 )

    print("\n===\n")

    print("* Ancilla added to a channel")
    c = hadamard
    s = random_state([2])
    print( (c @ ket(1).as_chan()) >> s == (c >> s) @ ket(1) )

    print("\n===\n")

    print("* Discard and ancilla")
    s = random_state([2])
    print( (discard([2]) @ ket(0).as_chan()) >> s )
    print( (ket(0).as_chan() @ discard([2])) >> s )

    print("\n===\n")

    print("* Ancilla, init and validity")
    dom = [2,5]
    w = random_state(dom)
    print( w == w.as_chan() >> init_state )
    p = random_pred(dom)
    print( w >= p )
    print( w.as_chan() << p )

    print("\n===\n")

    print("* Predicate subchannel equivalents")
    print( p == p.as_subchan() << truth([]) )
    print( np.all(np.isclose((p.as_subchan() >> w).array,
                             (w.as_chan() << p).array)) )

def measurement():
    
    print("\nSubsection: Measurement, control and instruments\n")

    print("* Probabilistic outcome of measurement")
    p = random_pred([5])
    s = random_state([5])
    print( s >= p )
    print( meas_pred(p) >> s )
    print("Predicate and it orthosupplement can be recovered: ",
          p == meas_pred(p) << yes_pred,
          ~p == meas_pred(p) << no_pred )

    print("\n===\n")

    print("* Quantum coin")
    print( meas0 >> (hadamard >> ket(0)) )
    print( (meas0 * hadamard) >> ket(1) )

    print("\n===\n")

    print("* Classic channel taking out the diagonal")
    s = random_state([3])
    print( s )
    print( classic([3]) >> s )

    print("\n===\n")

    print("* Measurement in the Bell basis")
    bell_test = [bell00.as_pred(), bell01.as_pred(), bell10.as_pred(), bell11.as_pred()]
    meas_bell = meas_test(bell_test)
    w = cnot >> random_state([2,2])
    print( w )
    print( meas_bell >> w )
    print( w >= bell00.as_pred() )
    print( w >= bell01.as_pred() )
    print( w >= bell10.as_pred() )
    print( w >= bell11.as_pred() )

    print("\n===\n")

    print("* Equality of channels")
    print( cnot * (classic([2]) @ idn([2])) == ccontrol(x_chan) )

    print("\n===\n")

    print("* ccase illustration")
    s = probabilistic_state(0.2, 0.3, 0.5)
    t = random_state([2])
    w = ccase(x_chan, hadamard, idn([2])) >> (s @ t)
    print( w % [1, 0] )
    print( w % [0, 1] )
    print( convex_state_sum((0.2, x_chan >> t), \
                            (0.3, hadamard >> t), \
                            (0.5, idn([2]) >> t)) )

    print("\n===\n")

    print("* First projection of instrument is measurement")
    p = random_pred([5])
    print( (idn([2]) @ discard([5])) * instr(p) == meas_pred(p) )

    print("\n===\n")

    print("* Second projection of instrument is convex sum")
    s = random_state([5])
    print( discard([2]) @ idn([5]) * instr(p) >> s ==
           convex_state_sum( (s >= p, s / p), (s >= ~p, s / ~p) ) )

    print("\n===\n")

    print("* Intrument predicate transformation")
    p = random_pred([10])
    q = random_pred([10])
    print( instr(p) << truth([2]) @ q == (p & q) + (~p & q) )
    print( instr(p) << point_pred(0,2) @ q == p & q )
    print( instr(p) << point_pred(1,2) @ q == ~p & q )

    print("\n===\n")

    print("* Predicate case")
    p = random_pred([2])
    s = random_state([2])
    print( pcase(p)(x_chan, y_chan) >> s ==
           convex_state_sum( (s >= p, x_chan >> s/p), (s >= ~p, y_chan >> s/~p) ) )
    q = random_pred([2])
    print( pcase(p)(x_chan, y_chan) << q ==
           (p & x_chan << q) + (~p & y_chan << q) )


    print("\n===\n")

    print("* Extract update state from instrument")
    p = random_pred([10])
    s = random_state([10])
    print( ((instr(p) >> s) / (point_pred(0,2) @ truth([10]))) % [0,1] ==  s / p )


def teleportation_and_superdensecoding():
    
    print("\nSection: Measurement, control and instruments\n")

    print("* Teleportation, with the Bell state")
    alice = (meas0 @ meas0) * (hadamard @ idn([2])) * cnot
    bob = (discard([2]) @ idn([2])) \
          * ccontrol(z_chan) \
          * (idn([2]) @ discard([2]) @ idn([2])) \
          * (idn([2]) @ ccontrol(x_chan))
    teleportation = bob * (alice @ idn([2])) * (idn([2]) @ bell00.as_chan())
    s = random_state([2])
    print( s )
    print( teleportation >> s )

    print("\n===\n")

    print("* Equality of states")
    print( s == (teleportation >> s) )

    print("\n===\n")

    print("* Equality of channels")
    print( np.isclose(teleportation.array, idn([2]).array) )
    print( np.all(np.isclose(teleportation.array, idn([2]).array)) )
    print( teleportation == idn([2]) )

    print("\n===\n")

    print("* Superdense coding, with the Bell state")
    alice = (discard([2]) @ idn([2])) * ccontrol(x_chan) \
            * (idn([2]) @ discard([2]) @ idn([2])) \
            * (idn([2]) @ ccontrol(z_chan)) * (swap @ idn([2]))
    bob = (meas0 @ meas0) * (hadamard @ idn([2])) * cnot
    def superdense_coding(r, s):
        return bob >> ((alice @ idn([2])) >> (cflip(r) @ cflip(s) @ bell00))
    r = random.uniform(0,1)
    s = random.uniform(0,1)
    print( r )
    print( s )
    print( superdense_coding(r,s) % [1,0] )
    print( superdense_coding(r,s) % [0,1] )

    print("\n===\n")

    print("* Superdense coding, with the Bell state, as channel")
    sdc = bob * (alice @ idn([2])) * (idn([2,2]) @ bell00.as_chan())
    sdck = kron(2,2) * sdc * kron_inv(2,2)
    print("channel types", sdc.dom, sdc.cod )
    print( sdc == classic([2,2]), sdck == classic([4]),  )

    print("\n===\n")

    print("* Teleportation, with the GHZ state")
    bell_test = [bell00.as_pred(), bell01.as_pred(), bell10.as_pred(), bell11.as_pred()]
    meas_bell = meas_test(bell_test)
    print( meas_bell.dom, meas_bell.cod )
    alice = (meas_bell @ idn([2,2])) * (idn([2]) @ ghz.as_chan())
    print( alice.dom, alice.cod )

    print("\n===\n")

    print("* GHZ-teleportation, Bob's side")
    hadamard_test = [plus.as_pred(), minus.as_pred()]
    meas_hadamard = meas_test(hadamard_test)
    bob = ( discard([8]) @ idn([2]) ) \
          * ccase(idn([2]), z_chan, x_chan, x_chan * z_chan, \
                  z_chan, idn([2]), x_chan * z_chan, x_chan) \
        * ( kron(4, 2) @ idn([2]) ) \
        * ( idn([4]) @ meas_hadamard @ idn([2]) )
    print( bob.dom, bob.cod )
    ghz_teleporation = bob * alice

    print("\n===\n")

    print("* GHZ-teleportation, test")
    s = random_state([2])
    print( s )
    print( ghz_teleporation >> s )
    print( ghz_teleporation == idn([2]) )

    print("\n===\n")

    print("* GHZ-superdense coding")
    iy_matrix = np.array([[0,1],
                          [-1,0]])
    iy_chan = channel_from_isometry(iy_matrix, [2], [2])
    alice = (discard([8]) @ idn([2,2])) * ccase(idn([2]) @ idn([2]), 
                                                idn([2]) @ x_chan, 
                                                x_chan @ idn([2]), 
                                                x_chan @ x_chan, 
                                                z_chan @ idn([2]), 
                                                z_chan @ x_chan, 
                                                iy_chan @ idn([2]), 
                                                iy_chan @ x_chan) 
    print( alice.dom, alice.cod )

    print("\n===\n")

    print("* GHZ-superdense coding, Bob's side")
    bob = meas_ghz
    ghz_sdc = bob * (alice @ idn([2])) * (idn([8]) @ ghz.as_chan())
    print( ghz_sdc.dom, ghz_sdc.cod )

    print("\n===\n")

    print("* GHZ-superdense coding 8-test")
    s = random_probabilistic_state([8])
    print( s )
    print( ghz_sdc >> s )

    print("\n===\n")

    print("* GHZ-superdense coding classical bit test")
    k1 = kron(4,2) * (kron(2,2) @ idn([2]))
    k2 = (kron_inv(2,2) @ idn([2])) * kron_inv(4,2)
    r1 = random_probabilistic_state([2])
    r2 = random_probabilistic_state([2])
    r3 = random_probabilistic_state([2])
    print( r1 )
    print( r2 )
    print( r3 )
    r = k1 >> (r1 @ r2 @ r3)
    w = k2 >> (ghz_sdc >> r)
    print( w % [1,0,0] )
    print( w % [0,1,0] )
    print( w % [0,0,1] )

def order_inference():

    print("\nSection: Order inference effect example\n")    



def all():
    states()
    operations_on_states()
    basic_states()
    predicates()
    operations_on_predicates()
    validity()
    conditioning()
    random_variables()
    state_transformation()
    predicate_transformation()
    structural_channels()
    measurement()
    teleportation_and_superdensecoding()
    # order_inference()


def main():
    all()
    # states()
    # operations_on_states()
    # basic_states()
    # predicates()
    # operations_on_predicates()
    # validity()
    # conditioning()
    # random_variables()
    # state_transformation()
    # predicate_transformation()
    # structural_channels()
    # measurement()
    # teleportation_and_superdensecoding()
    # order_inference()

if __name__ == "__main__":
    main()
