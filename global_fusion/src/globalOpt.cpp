/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "globalOpt.h"
#include "Factors.h"

GlobalOptimization::GlobalOptimization()
{
	initGPS = false;
    newGPS = false;
	WGPS_T_WVIO = Eigen::Matrix4d::Identity();
    threadOpt = std::thread(&GlobalOptimization::optimize, this);
}

GlobalOptimization::~GlobalOptimization()
{   
    printf("GlobalOptimization deconstructor is called\n");
}

void GlobalOptimization::shutdown() {
    printf("GlobalOptimization shutdown is called\n");
    shutting_down = true;
    cvGPS.notify_one();
    if (threadOpt.joinable())
        threadOpt.join();
}

void GlobalOptimization::GPS2XYZ(double latitude, double longitude, double altitude, double* xyz)
{
    if(!initGPS)
    {
        geoConverter.Reset(latitude, longitude, altitude);
        initGPS = true;
    }
    geoConverter.Forward(latitude, longitude, altitude, xyz[0], xyz[1], xyz[2]);
    //printf("la: %f lo: %f al: %f\n", latitude, longitude, altitude);
    //printf("gps x: %f y: %f z: %f\n", xyz[0], xyz[1], xyz[2]);
}

void GlobalOptimization::inputOdom(double t, Eigen::Vector3d OdomP, Eigen::Quaterniond OdomQ)
{
	mPoseMap.lock();
    vector<double> localPose{OdomP.x(), OdomP.y(), OdomP.z(), 
    					     OdomQ.w(), OdomQ.x(), OdomQ.y(), OdomQ.z()};
    localPoseMap[t] = localPose;


    Eigen::Quaterniond globalQ;
    globalQ = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomQ;
    Eigen::Vector3d globalP = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomP + WGPS_T_WVIO.block<3, 1>(0, 3);
    vector<double> globalPose{globalP.x(), globalP.y(), globalP.z(),
                              globalQ.w(), globalQ.x(), globalQ.y(), globalQ.z()};
    globalPoseMap[t] = globalPose;
    lastP = globalP;
    lastQ = globalQ;

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(t);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = lastP.x();
    pose_stamped.pose.position.y = lastP.y();
    pose_stamped.pose.position.z = lastP.z();
    pose_stamped.pose.orientation.x = lastQ.x();
    pose_stamped.pose.orientation.y = lastQ.y();
    pose_stamped.pose.orientation.z = lastQ.z();
    pose_stamped.pose.orientation.w = lastQ.w();
    global_path.header = pose_stamped.header;
    global_path.poses.push_back(pose_stamped);

    mPoseMap.unlock();
}

void GlobalOptimization::getGlobalOdom(Eigen::Vector3d &odomP, Eigen::Quaterniond &odomQ)
{
    odomP = lastP;
    odomQ = lastQ;
}

void GlobalOptimization::inputGPS(double t, double latitude, double longitude, double altitude, double posAccuracy)
{
	double xyz[3];
	GPS2XYZ(latitude, longitude, altitude, xyz);
	vector<double> tmp{xyz[0], xyz[1], xyz[2], posAccuracy};
    //printf("new gps: t: %f x: %f y: %f z:%f \n", t, tmp[0], tmp[1], tmp[2]);
    
    {
        std::lock_guard<std::mutex> lk(mtxGPS);
        GPSPositionMap[t] = tmp;
        newGPS = true;
    }
    cvGPS.notify_one();

}

void GlobalOptimization::optimize()
{   
    while(true)
    {   
        // ---- 1) Wait for a new GPS measurement to trigger an optimization ----
        // Wait until inputGPS() has flipped newGPS to true
        std::unique_lock<std::mutex> lk(mtxGPS);
        cvGPS.wait(lk, [this]{ return newGPS || shutting_down; });
        if (shutting_down) break;
        newGPS = false;   // reset the flag
        lk.unlock();

        // ROS_INFO("Global optimization");
        printf("Global optimization\n");
        TicToc globalOptimizationTime;

        // ---- 2) Set up your Ceres problem ----
        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        //options.minimizer_progress_to_stdout = true;
        //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
        options.max_num_iterations = 5;
        ceres::Solver::Summary summary;
        // ceres::LossFunction *loss_function;
        // loss_function = new ceres::HuberLoss(1.0);
        auto loss_function = new ceres::HuberLoss(1.0);
        ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();

        // ---- 3) Lock and snapshot all stored poses ----
        //add param
        mPoseMap.lock();
        int length = localPoseMap.size();
        // Pre‐allocate arrays for poses
        // w^t_i   w^q_i
        double t_array[length][3];
        double q_array[length][4];

        // Fill arrays from the current global poses
        map<double, vector<double>>::iterator iter;
        iter = globalPoseMap.begin();
        for (int i = 0; i < length; i++, iter++)
        {   
            // position
            t_array[i][0] = iter->second[0];
            t_array[i][1] = iter->second[1];
            t_array[i][2] = iter->second[2];
            // orientation quaternion
            q_array[i][0] = iter->second[3];
            q_array[i][1] = iter->second[4];
            q_array[i][2] = iter->second[5];
            q_array[i][3] = iter->second[6];

            // Tell Ceres these are optimization variables
            problem.AddParameterBlock(q_array[i], 4, local_parameterization);
            problem.AddParameterBlock(t_array[i], 3);
        }

        // ---- 4) Add VIO relative‐pose factors between consecutive timestamps ----
        map<double, vector<double>>::iterator iterVIO, iterVIONext, iterGPS;
        int i = 0;
        for (iterVIO = localPoseMap.begin(); iterVIO != localPoseMap.end(); iterVIO++, i++)
        {
            //vio factor
            iterVIONext = iterVIO;
            iterVIONext++;
            if(iterVIONext != localPoseMap.end())
            {
                // Build the “measured” relative transform from VIO
                Eigen::Matrix4d wTi = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d wTj = Eigen::Matrix4d::Identity();
                // fill wTi from lit->second
                wTi.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIO->second[3], iterVIO->second[4], 
                                                            iterVIO->second[5], iterVIO->second[6]).toRotationMatrix();
                wTi.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIO->second[0], iterVIO->second[1], iterVIO->second[2]);
                
                // fill wTj from next->second
                wTj.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIONext->second[3], iterVIONext->second[4], 
                                                            iterVIONext->second[5], iterVIONext->second[6]).toRotationMatrix();
                wTj.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIONext->second[0], iterVIONext->second[1], iterVIONext->second[2]);
                
                // Compute relative transform i→j
                Eigen::Matrix4d iTj = wTi.inverse() * wTj;
                Eigen::Quaterniond iQj;
                iQj = iTj.block<3, 3>(0, 0);
                Eigen::Vector3d iPj = iTj.block<3, 1>(0, 3);
                
                // Create and add the Ceres cost‐function for VIO
                ceres::CostFunction* vio_function = RelativeRTError::Create(iPj.x(), iPj.y(), iPj.z(),
                                                                            iQj.w(), iQj.x(), iQj.y(), iQj.z(),
                                                                            /*noise=*/1, /*quat_noise=*/0.1);
                problem.AddResidualBlock(vio_function,  /*loss=*/nullptr, q_array[i], t_array[i], q_array[i+1], t_array[i+1]);

                /*
                double **para = new double *[4];
                para[0] = q_array[i];
                para[1] = t_array[i];
                para[3] = q_array[i+1];
                para[4] = t_array[i+1];

                double *tmp_r = new double[6];
                double **jaco = new double *[4];
                jaco[0] = new double[6 * 4];
                jaco[1] = new double[6 * 3];
                jaco[2] = new double[6 * 4];
                jaco[3] = new double[6 * 3];
                vio_function->Evaluate(para, tmp_r, jaco);

                std::cout << Eigen::Map<Eigen::Matrix<double, 6, 1>>(tmp_r).transpose() << std::endl
                    << std::endl;
                std::cout << Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>>(jaco[0]) << std::endl
                    << std::endl;
                std::cout << Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>>(jaco[1]) << std::endl
                    << std::endl;
                std::cout << Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>>(jaco[2]) << std::endl
                    << std::endl;
                std::cout << Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>>(jaco[3]) << std::endl
                    << std::endl;
                */

            }

            // ---- 5) Add GPS absolute‐position factors where available ----
            //gps factor
            double t = iterVIO->first;
            iterGPS = GPSPositionMap.find(t);
            if (iterGPS != GPSPositionMap.end())
            {
                // xyz + covariance weight
                ceres::CostFunction* gps_function = TError::Create(iterGPS->second[0], iterGPS->second[1], 
                                                                    iterGPS->second[2], iterGPS->second[3]);
                //printf("inverse weight %f \n", iterGPS->second[3]);
                // loss_function: robust Huber on GPS
                // t_array[i]: only the position block
                problem.AddResidualBlock(gps_function, loss_function, t_array[i]);

                /*
                double **para = new double *[1];
                para[0] = t_array[i];

                double *tmp_r = new double[3];
                double **jaco = new double *[1];
                jaco[0] = new double[3 * 3];
                gps_function->Evaluate(para, tmp_r, jaco);

                std::cout << Eigen::Map<Eigen::Matrix<double, 3, 1>>(tmp_r).transpose() << std::endl
                    << std::endl;
                std::cout << Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(jaco[0]) << std::endl
                    << std::endl;
                */
            }

        }

        // ---- 6) Solve the entire batch ----
        //mPoseMap.unlock();
        ceres::Solve(options, &problem, &summary);
        //std::cout << summary.BriefReport() << "\n";

        // ---- 7) Write back optimized poses & update frame‐alignment ----
        // update global pose
        //mPoseMap.lock();
        iter = globalPoseMap.begin();
        for (int i = 0; i < length; i++, iter++)
        {
            vector<double> globalPose{t_array[i][0], t_array[i][1], t_array[i][2],
                                        q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]};
            iter->second = globalPose;
            if(i == length - 1)
            {
                Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity(); 
                Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();
                double t = iter->first;
                WVIO_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4], 
                                                                    localPoseMap[t][5], localPoseMap[t][6]).toRotationMatrix();
                WVIO_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
                WGPS_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(globalPose[3], globalPose[4], 
                                                                    globalPose[5], globalPose[6]).toRotationMatrix();
                WGPS_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);
                WGPS_T_WVIO = WGPS_T_body * WVIO_T_body.inverse();
            }
        }
        updateGlobalPath();
        printf("global time %f \n", globalOptimizationTime.toc());
        // ROS_INFO("global time %f ", globalOptimizationTime.toc());
        mPoseMap.unlock();
        
        lk.lock();
    }
	return;
}


void GlobalOptimization::updateGlobalPath()
{
    global_path.poses.clear();
    map<double, vector<double>>::iterator iter;
    for (iter = globalPoseMap.begin(); iter != globalPoseMap.end(); iter++)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time(iter->first);
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose.position.x = iter->second[0];
        pose_stamped.pose.position.y = iter->second[1];
        pose_stamped.pose.position.z = iter->second[2];
        pose_stamped.pose.orientation.w = iter->second[3];
        pose_stamped.pose.orientation.x = iter->second[4];
        pose_stamped.pose.orientation.y = iter->second[5];
        pose_stamped.pose.orientation.z = iter->second[6];
        global_path.poses.push_back(pose_stamped);
    }
}