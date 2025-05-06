#pragma once
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp/clock.hpp"
#include "builtin_interfaces/msg/time.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <nav_msgs/msg/path.hpp>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <sensor_msgs/msg/imu.h>

#include "tf2/transform_datatypes.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>   
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "rmw/types.h"
#include "rmw/qos_profiles.h"


//@@ extra inclusion
#include "string.h"
#include <shared_mutex>
#include <eigen3/Eigen/Dense>
#include <vector>
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <cassert>
typedef Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> MapMatrixCol;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixXdRow;
using AXXf = Eigen::ArrayXXf;
using AXf = Eigen::ArrayXf;       // column vector
using PointXYZIPtr = pcl::PointCloud<pcl::PointXYZI>::Ptr;
using PointXYPtr   = pcl::PointCloud<pcl::PointXY>::Ptr;
using KdTreePtr    = pcl::search::KdTree<pcl::PointXY>::Ptr;

using namespace std;
const double PI = 3.1415926;

#define PLOTPATHSET 1


class ScanPreProcess:public rclcpp::Node{
    public:
        
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud;
        rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr subGoal;
        rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subSpeed;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubFreePaths;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_obs_;
        //@@ parameters and default values
        int dim_control = 2;
        int dim_orientation = 3; // r,p,y
        int dim_coord = 3; // x,y,z
        AXf vehicle_orien; 
        //float vehicleX = 0, vehicleY = 0, vehicleZ = 0;
        AXf vehicle_coord;
        AXf goal_coord;
        string pathFolder;
        double vehicleLength = 0.6;
        double vehicleWidth = 0.6;
        float disc_radius = 0.3 + 0.05; // 0.05 is buffer 
        float step_length = pow(2,0.5)*disc_radius;
        float cluster_tolerance = 0.1; // 0.1
        double sensorOffsetX = 0;
        double sensorOffsetY = 0;
        bool twoWayDrive = true;
        double laserVoxelSize = 0.05;
        double terrainVoxelSize = 0.2;
        bool useTerrainAnalysis = false;
        bool checkObstacle = true;
        bool checkRotObstacle = false;
        double adjacentRange = 3.5;
        double obstacleHeightThre = 0.2;
        double groundHeightThre = 0.1;
        double costHeightThre = 0.1;
        double costScore = 0.02;
        bool useCost = false;
        const int laserCloudStackNum = 1;
        int laserCloudCount = 0;
        int pointPerPathThre = 2;
        double minRelZ = -0.5;
        double maxRelZ = 0.25;
        double maxSpeed = 1.0;
        double dirWeight = 0.02;
        double dirThre = 90.0;
        bool dirToVehicle = false;
        double pathScale = 1.0;
        double minPathScale = 0.75;
        double pathScaleStep = 0.25;
        bool pathScaleBySpeed = true;
        double minPathRange = 1.0;
        double pathRangeStep = 0.5;
        bool pathRangeBySpeed = true;
        bool pathCropByGoal = true;
        bool autonomyMode = false;
        double autonomySpeed = 1.0;
        double joyToSpeedDelay = 2.0;
        double joyToCheckObstacleDelay = 5.0;
        double goalClearRange = 0.5;

        float joySpeed = 0;
        float joySpeedRaw = 0;
        float joyDir = 0;
        float goalX = 0.0;
        float goalY = 0.0;
        const int pathNum = 343;
        const int groupNum = 7;
        float gridVoxelSize = 0.02;
        float searchRadius = 0.45;
        float gridVoxelOffsetX = 3.2;
        float gridVoxelOffsetY = 4.5;
        const int gridVoxelNumX = 161;
        const int gridVoxelNumY = 451;
        const int gridVoxelNum = gridVoxelNumX * gridVoxelNumY;

        PointXYZIPtr laserCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        PointXYZIPtr laserCloudCrop = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        PointXYPtr laserCloudCrop2d = std::make_shared<pcl::PointCloud<pcl::PointXY>>();
        pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudDwz = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        pcl::search::KdTree<pcl::PointXY>::Ptr tree = std::make_shared<pcl::search::KdTree<pcl::PointXY>>();
        PointXYZIPtr plannerCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        PointXYZIPtr plannerCloudCrop = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        pcl::VoxelGrid<pcl::PointXYZI> laserDwzFilter;
        
        // pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloud(new pcl::PointCloud<pcl::PointXYZI>());
        //pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCrop(new pcl::PointCloud<pcl::PointXYZI>());
        //pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudDwz(new pcl::PointCloud<pcl::PointXYZI>());
        


        // pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloud(new pcl::PointCloud<pcl::PointXYZI>());
        // pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloudCrop(new pcl::PointCloud<pcl::PointXYZI>());
        // pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloudDwz(new pcl::PointCloud<pcl::PointXYZI>());
        // pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudStack[laserCloudStackNum];
        // pcl::PointCloud<pcl::PointXYZI>::Ptr plannerCloud(new pcl::PointCloud<pcl::PointXYZI>());
        // pcl::PointCloud<pcl::PointXYZI>::Ptr plannerCloudCrop(new pcl::PointCloud<pcl::PointXYZI>());
        // pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryCloud(new pcl::PointCloud<pcl::PointXYZI>());
        // pcl::PointCloud<pcl::PointXYZI>::Ptr addedObstacles(new pcl::PointCloud<pcl::PointXYZI>());
        // pcl::PointCloud<pcl::PointXYZ>::Ptr startPaths[groupNum];
        // #if PLOTPATHSET == 1
        // pcl::PointCloud<pcl::PointXYZI>::Ptr paths[pathNum];
        // pcl::PointCloud<pcl::PointXYZI>::Ptr freePaths = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        // #endif

        // int pathList[pathNum] = {0};
        // float endDirPathList[pathNum] = {0};
        // int clearPathList[36 * pathNum] = {0};
        // float pathPenaltyList[36 * pathNum] = {0};
        // float clearPathPerGroupScore[36 * groupNum] = {0};
        // std::vector<int> correspondences[gridVoxelNum];

        bool newLaserCloud = false;
        bool newTerrainCloud = false;

        double odomTime = 0;
        double joyTime = 0;

        //float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0;

        //float vehicleX = 0, vehicleY = 0, vehicleZ = 0;


        


        // @@ definations of the callback and utility functions
        ScanPreProcess();
        void on_activate();
        void laserCloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr );
        void extractObstacles();
        void odometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr);
        void goalHandler(const geometry_msgs::msg::PointStamped::ConstSharedPtr );
        std::shared_mutex cloud_mutex;
};