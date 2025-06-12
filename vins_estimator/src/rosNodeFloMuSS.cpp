#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

Estimator estimator;
std::mutex m_buf;

// Global variables for stereo mode and synchronization
bool current_stereo = STEREO;
std::mutex stereo_mutex;
double GPS_SYNC_TOLERANCE = -1;
std::string OUTPUT_FOLDER;
double last_header_ts = -1;
double last_time = -1;

cv::Mat getImageFromCompressedMsg(const sensor_msgs::CompressedImageConstPtr &img_msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        return cv_ptr->image.clone();
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return cv::Mat();
    }
}

// Stereo callback for synchronized images
void stereo_callback(const sensor_msgs::CompressedImageConstPtr &img0_msg,
                     const sensor_msgs::CompressedImageConstPtr &img1_msg) {
    bool stereo_mode;
    {
        std::lock_guard<std::mutex> lock(stereo_mutex);
        stereo_mode = current_stereo;
    }
    
    if (!stereo_mode) return;

    cv::Mat image0 = getImageFromCompressedMsg(img0_msg);
    cv::Mat image1 = getImageFromCompressedMsg(img1_msg);

    ROS_DEBUG_STREAM("Received stereo images at time: " << img0_msg->header.stamp.toSec());
    // if (last_header_ts < 0){
    //     last_header_ts = img0_msg->header.stamp.toSec();
    // }else{
    //     ROS_INFO("At callback time difference: %3f ms", (img0_msg->header.stamp.toSec() - last_header_ts) * 1000 );
    //     last_header_ts = img0_msg->header.stamp.toSec();
    // }

    if (!image0.empty() && !image1.empty()) {
        double time = img0_msg->header.stamp.toSec();
        // if (last_time < 0){
        //     last_time = time;
        // }else{
        //     ROS_INFO("Before input image: %3f ms", (time - last_time) * 1000);
        //     last_time = time;
        // }
        // ROS_INFO("Before input image: %3f", time);
        // int width  = image0.cols;
        // int height = image0.rows;
        // std::cout << "Image size: " << width << " x " << height << std::endl;
        estimator.inputImage(time, image0, image1);
    }
}

// Mono callback for single image
void img0_callback(const sensor_msgs::CompressedImageConstPtr &img_msg) {
    bool stereo_mode;
    {
        std::lock_guard<std::mutex> lock(stereo_mutex);
        stereo_mode = current_stereo;
    }
    
    if (stereo_mode) return;

    cv::Mat image = getImageFromCompressedMsg(img_msg);
    if (!image.empty()) {
        double time = img_msg->header.stamp.toSec();
        estimator.inputImage(time, image);
    }
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg) {
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++) {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5) {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg) {
    if (restart_msg->data == true) {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg) {
    if (switch_msg->data == true) {
        estimator.changeSensorType(1, STEREO);
    } else {
        estimator.changeSensorType(0, STEREO);
    }
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg) {
    {
        std::lock_guard<std::mutex> lock(stereo_mutex);
        current_stereo = switch_msg->data;
    }
    
    if (switch_msg->data == true) {
        estimator.changeSensorType(USE_IMU, 1);
    } else {
        estimator.changeSensorType(USE_IMU, 0);
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 2) {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    readParameters(config_file);

    if (GPS_SYNC_TOLERANCE > 0){
        n.setParam("/globalEstimator/gps_sync_tolerance", GPS_SYNC_TOLERANCE);
        ROS_INFO("Setting GPS_SYNC_TOLERANCE = %f", GPS_SYNC_TOLERANCE);
    }
    n.setParam("/globalEstimator/output_path", OUTPUT_FOLDER);

    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    // Initialize stereo mode
    current_stereo = STEREO;

    // Setup subscribers
    ros::Subscriber sub_imu;
    if(USE_IMU) {
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);

    // Setup message filters for stereo
    if(STEREO) {
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_img0_sync(n, IMAGE0_TOPIC, 100);
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_img1_sync(n, IMAGE1_TOPIC, 100);

        typedef message_filters::sync_policies::ApproximateTime<
            sensor_msgs::CompressedImage, 
            sensor_msgs::CompressedImage> SyncPolicy;
        
        // Set synchronization tolerance to 0.003 seconds
        SyncPolicy sync_policy(10);
        sync_policy.setMaxIntervalDuration(ros::Duration(0.003));
        
        message_filters::Synchronizer<SyncPolicy> sync(sync_policy);
        sync.connectInput(sub_img0_sync, sub_img1_sync);
        sync.registerCallback(boost::bind(&stereo_callback, _1, _2));

        ros::spin();
    } else {
        ros::spin();
    }

    return 0;
}