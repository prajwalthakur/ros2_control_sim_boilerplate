#include "rclcpp/rclcpp.hpp"
#include "local_preproc_cpp/scan_preprocess.hpp"
//auto node = std::make_shared<ScanPreProcess>();  // Use your custom class
ScanPreProcess::ScanPreProcess():Node("scan_preprocess_node") {
        vehicle_orien.resize(dim_orientation);
        vehicle_coord.resize(dim_coord);
        goal_coord.resize(dim_coord);
        // @@ declare parameters
        this->declare_parameter<std::string>("pathFolder", pathFolder);
        this->declare_parameter<double>("vehicleLength", vehicleLength);
        this->declare_parameter<double>("vehicleWidth", vehicleWidth);
        this->declare_parameter<double>("sensorOffsetX", sensorOffsetX);
        this->declare_parameter<double>("sensorOffsetY", sensorOffsetY);
        this->declare_parameter<bool>("twoWayDrive", twoWayDrive);
        this->declare_parameter<double>("laserVoxelSize", laserVoxelSize);
        this->declare_parameter<double>("terrainVoxelSize", terrainVoxelSize);
        this->declare_parameter<bool>("useTerrainAnalysis", useTerrainAnalysis);
        this->declare_parameter<bool>("checkObstacle", checkObstacle);
        this->declare_parameter<bool>("checkRotObstacle", checkRotObstacle);
        this->declare_parameter<double>("adjacentRange", adjacentRange);
        this->declare_parameter<double>("obstacleHeightThre", obstacleHeightThre);
        this->declare_parameter<double>("groundHeightThre", groundHeightThre);
        this->declare_parameter<double>("costHeightThre", costHeightThre);
        this->declare_parameter<double>("costScore", costScore);
        this->declare_parameter<bool>("useCost", useCost);
        this->declare_parameter<int>("pointPerPathThre", pointPerPathThre);
        this->declare_parameter<double>("minRelZ", minRelZ);
        this->declare_parameter<double>("maxRelZ", maxRelZ);
        this->declare_parameter<double>("maxSpeed", maxSpeed);
        this->declare_parameter<double>("dirWeight", dirWeight);
        this->declare_parameter<double>("dirThre", dirThre);
        this->declare_parameter<bool>("dirToVehicle", dirToVehicle);
        this->declare_parameter<double>("pathScale", pathScale);
        this->declare_parameter<double>("minPathScale", minPathScale);
        this->declare_parameter<double>("pathScaleStep", pathScaleStep);
        this->declare_parameter<bool>("pathScaleBySpeed", pathScaleBySpeed);
        this->declare_parameter<double>("minPathRange", minPathRange);
        this->declare_parameter<double>("pathRangeStep", pathRangeStep);
        this->declare_parameter<bool>("pathRangeBySpeed", pathRangeBySpeed);
        this->declare_parameter<bool>("pathCropByGoal", pathCropByGoal);
        this->declare_parameter<bool>("autonomyMode", autonomyMode);
        this->declare_parameter<double>("autonomySpeed", autonomySpeed);
        this->declare_parameter<double>("joyToSpeedDelay", joyToSpeedDelay);
        this->declare_parameter<double>("joyToCheckObstacleDelay", joyToCheckObstacleDelay);
        this->declare_parameter<double>("goalClearRange", goalClearRange);
        this->declare_parameter<double>("goalX", goalX);
        this->declare_parameter<double>("goalY", goalY);

        // @@ get the parameters from ros2 server
       this->get_parameter("pathFolder", pathFolder);
       this->get_parameter("vehicleLength", vehicleLength);
       this->get_parameter("vehicleWidth", vehicleWidth);
       this->get_parameter("sensorOffsetX", sensorOffsetX);
       this->get_parameter("sensorOffsetY", sensorOffsetY);
       this->get_parameter("twoWayDrive", twoWayDrive);
       this->get_parameter("laserVoxelSize", laserVoxelSize);
       this->get_parameter("terrainVoxelSize", terrainVoxelSize);
       this->get_parameter("useTerrainAnalysis", useTerrainAnalysis);
       this->get_parameter("checkObstacle", checkObstacle);
       this->get_parameter("checkRotObstacle", checkRotObstacle);
       this->get_parameter("adjacentRange", adjacentRange);
       this->get_parameter("obstacleHeightThre", obstacleHeightThre);
       this->get_parameter("groundHeightThre", groundHeightThre);
       this->get_parameter("costHeightThre", costHeightThre);
       this->get_parameter("costScore", costScore);
       this->get_parameter("useCost", useCost);
       this->get_parameter("pointPerPathThre", pointPerPathThre);
       this->get_parameter("minRelZ", minRelZ);
       this->get_parameter("maxRelZ", maxRelZ);
       this->get_parameter("maxSpeed", maxSpeed);
       this->get_parameter("dirWeight", dirWeight);
       this->get_parameter("dirThre", dirThre);
       this->get_parameter("dirToVehicle", dirToVehicle);
       this->get_parameter("pathScale", pathScale);
       this->get_parameter("minPathScale", minPathScale);
       this->get_parameter("pathScaleStep", pathScaleStep);
       this->get_parameter("pathScaleBySpeed", pathScaleBySpeed);
       this->get_parameter("minPathRange", minPathRange);
       this->get_parameter("pathRangeStep", pathRangeStep);
       this->get_parameter("pathRangeBySpeed", pathRangeBySpeed);
       this->get_parameter("pathCropByGoal", pathCropByGoal);
       this->get_parameter("autonomyMode", autonomyMode);
       this->get_parameter("autonomySpeed", autonomySpeed);
       this->get_parameter("joyToSpeedDelay", joyToSpeedDelay);
       this->get_parameter("joyToCheckObstacleDelay", joyToCheckObstacleDelay);
       this->get_parameter("goalClearRange", goalClearRange);
       this->get_parameter("goalX", goalX);
       this->get_parameter("goalY", goalY);


       //@@ initialization 
        laserDwzFilter.setLeafSize(laserVoxelSize, laserVoxelSize, laserVoxelSize);


}

void ScanPreProcess::on_activate(){
    // vehicle_ = std::make_shared<VehicleClass>(shared_from_this());
    //@@ declare subscriber and publisher
    subOdometry = this->create_subscription<nav_msgs::msg::Odometry>(
        "/state_estimation", 5,
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
            this->odometryHandler(msg);
        });

    subLaserCloud = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/registered_scan", 5,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            this->laserCloudHandler(msg);
        });

    subGoal = this->create_subscription<geometry_msgs::msg::PointStamped>(
        "/way_point", 5,
        [this](const geometry_msgs::msg::PointStamped::SharedPtr msg) {
            this->goalHandler(msg);
        });
    // subSpeed = this->create_subscription<std_msgs::msg::Float32>("/speed", 5, this->speedHandler);
    pubPath = this->create_publisher<nav_msgs::msg::Path>("/path", 5);
    pubFreePaths = this->create_publisher<sensor_msgs::msg::PointCloud2>("/free_paths", 2);
    pub_obs_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/obs_list", 2);

}

//@@ get the point cloud, filter out the point clouds whose distance is greater than some threshold distance
//@@ then convert those to the voxel grids
void ScanPreProcess::laserCloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr laser_msg){
    //@default useTerrainAnalysis:=faluse
    std::unique_lock lock(cloud_mutex);
    if(!useTerrainAnalysis){
        laserCloud->clear();
        pcl::fromROSMsg(*laser_msg, *laserCloud);
        pcl::PointXYZI point;
        laserCloudCrop->clear();
        laserCloudCrop2d->clear();
        int laser_cloud_size = laserCloud->points.size();
        //@@ calculate the distance of each cloud point from the vehicle and filter out the 
        // points which are the greater distance than defined adjacentRange
        AXf cloud_point_coord(this->dim_coord);
        pcl::PointXY q;
        for(int i=0;i<laser_cloud_size;++i){
            point  = laserCloud->points[i];
            cloud_point_coord<< point.x, point.y, point.z;
            q.x = point.x;
            q.y = point.y;
            float dis_from_veh = (cloud_point_coord-vehicle_coord).matrix().norm();
            if(dis_from_veh<adjacentRange){
                laserCloudCrop->points.push_back(point);
                laserCloudCrop2d->points.push_back(q);
            }
        }
        //@ conveer the point cloud to the voxel cubes
        laserCloudDwz->clear();
        laserDwzFilter.setInputCloud(laserCloudCrop);
        laserDwzFilter.filter(*laserCloudDwz);
        newLaserCloud = true;
        extractObstacles();
    }
}

void ScanPreProcess::extractObstacles(){
    if (laserCloudCrop2d->empty()) return;
    tree->setInputCloud(laserCloudCrop2d);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXY> ec;
    ec.setInputCloud(laserCloudCrop2d);
    ec.setSearchMethod(tree);

    /* magic numbers you tune */
    ec.setClusterTolerance( this->cluster_tolerance );   // metres – neighbours closer than this → same cluster
    ec.setMinClusterSize( 5 );       // drop single outliers
    ec.setMaxClusterSize( 5000 );    // (optional)

    ec.extract(cluster_indices);

    //@@ make circles around the clusters
    std::vector<Eigen::Vector2f> disc_centres;   // final result
    Eigen::Vector2f tmp;
    for(auto &indices : cluster_indices){
        float xmin = std::numeric_limits<float>::max();
        float xmax = std::numeric_limits<float>::lowest();
        float ymin = std::numeric_limits<float>::max();
        float ymax = std::numeric_limits<float>::lowest();
        for(auto &idx:indices.indices){
            pcl::PointXY pq = laserCloudCrop2d->points[idx];
            if(pq.x<xmin) xmin = pq.x;
            if(pq.x>xmax) xmax = pq.x;
            if(pq.y<ymin) ymin = pq.y;
            if(pq.y>ymax) ymax = pq.y;
        }
    
        for(auto x = xmin;x<xmax;x+=step_length){
            for(auto y= ymin; y < ymax; y+=step_length){
                bool touches = false;
                for(auto& index:indices.indices){
                    pcl::PointXY local_pq = laserCloudCrop2d->points[index];
                    if ( (local_pq.x - x)*(local_pq.x - x) + (local_pq.y - y)*(local_pq.y - y) < disc_radius*disc_radius ) {
                        touches = true;
                        break;
                    }
                }
                if(touches){
                    tmp<<x,y;
                    disc_centres.push_back(tmp);
                }

            }
        }
    }
    std_msgs::msg::Float32MultiArray arr;
    arr.layout.dim.resize(2);
    arr.layout.dim[0].label = "circles";
    arr.layout.dim[0].size = disc_centres.size();
    arr.layout.dim[1].label = "xy";
    arr.layout.dim[1].size  = 2;
    
    arr.data.reserve(disc_centres.size()*2);
    for (const auto &c : disc_centres) {
    arr.data.push_back(c.x());
    arr.data.push_back(c.y());
    }
    pub_obs_->publish(arr);
}

//@@ update the goal 
void ScanPreProcess::goalHandler(const geometry_msgs::msg::PointStamped::ConstSharedPtr goal_msg){
    goal_coord<<goal_msg->point.x, goal_msg->point.y,0.0;
}

//@@ update the vehicle coord
void ScanPreProcess::odometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr odom)
{
  odomTime = rclcpp::Time(odom->header.stamp).seconds();
  double roll, pitch, yaw;
  float x_coord, y_coord, z_coord;
  geometry_msgs::msg::Quaternion geoQuat = odom->pose.pose.orientation;
  tf2::Matrix3x3(tf2::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);
  float rl = static_cast<float>(roll);
  float pt = static_cast<float>(pitch);
  float yw = static_cast<float> (yaw);
  vehicle_orien<< rl, pt, yw;
  x_coord = odom->pose.pose.position.x - cos(yaw) * sensorOffsetX + sin(yaw) * sensorOffsetY;
  y_coord = odom->pose.pose.position.y - sin(yaw) * sensorOffsetX - cos(yaw) * sensorOffsetY;
  z_coord = odom->pose.pose.position.z;
  vehicle_coord<<x_coord, y_coord, z_coord;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ScanPreProcess>();  
    node->on_activate();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


