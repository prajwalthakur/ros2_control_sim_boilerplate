                ┌─────────────┐
/registered_scan│  C++ node   │ (laserCloudHandler from the package you showed)
/odom──────────►│  Pre‑proc   │──►  /mppi/obstacles  (compact numpy array)
/way_point─────►│             │
                └─────────────┘
                                 DDS
                ┌─────────────────────────────────────┐
                │  Python rclpy node  (your MPPI)     │
/mppi/obstacles │  ● subscribes Odometry, Waypoint,   │
/state_estimation│    obstacle array                  │
                │  ● runs MPPI loop @ ≈ 10–20 Hz      │
                │  ● publishes geometry_msgs/Twist   │
                └─────────────────────────────────────┘
                
