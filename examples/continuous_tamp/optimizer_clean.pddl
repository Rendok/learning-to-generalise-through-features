(define (stream pick-and-place)
  (:function (Distance ?q1 ?q2)
    (and (Conf ?q1) (Conf ?q2))
  )
  (:predicate (TrajCollision ?t ?b2 ?p2)
    (and (Traj ?t) (Pose ?b2 ?p2))
  )
  (:optimizer gurobi

    ; Constructs a set of free variables
    (:variable ?p
      :inputs (?b ?r)
      :domain (Placeable ?b ?r)
      :graph (and (Contained ?b ?p ?r) (Pose ?b ?p)))
    (:variable ?q
      :graph (Conf ?q))

    ; Constraint forms that can be optimized
    (:constraint (Contained ?b ?p ?r)
      :necessary (and (Placeable ?b ?r) (Pose ?b ?p)))
    (:constraint (Kin ?b ?q ?p)
      :necessary (and (Pose ?b ?p) (Conf ?q)))
    (:constraint (CFree ?b1 ?p1 ?b2 ?p2)
      :necessary (and (Pose ?b1 ?p1) (Pose ?b2 ?p2)))

    ; Additive objective functions
    (:objective Distance)
  )

  (:optimizer rrt
    (:variable ?t
      :graph (Traj ?t))
    (:constraint (Motion ?q1 ?t ?q2)
      :necessary (and (Conf ?q1) (Traj ?t) (Conf ?q2)))

    ; Treating predicate as objective
    (:objective TrajCollision)
  )
)