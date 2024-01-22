#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <controller_interface/controller.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/WrenchStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <math.h>
#include <time.h>
#include <random>
#include <fstream>
#include <iomanip> 
#include <rosdyn_core/primitives.h>
#include <rosdyn_core/urdf_parser.h>
#include <comau_hw_interface/comau_hw_interface.h>
#include <comau_hw_interface/comau_msg.h>
// #include <eigen_matrix_utils/

#define N_JOINTS 6


namespace custom_position_controller {

  class MyPositionController : public controller_interface::Controller<hardware_interface::PositionJointInterface>
  {
    bool init(hardware_interface::PositionJointInterface* robot, ros::NodeHandle &nh) override
    {
    //Init cnr logger
    std::string n = "CONTROLLER" + nh.getNamespace();
    std::replace(n.begin(), n.end(), '/', '_');
    std::string what;
    if (!logger.init(n, "/comau_hw", false, false, &what))
    {
      std::cerr << __PRETTY_FUNCTION__ << ":" << "Error in creating the logger!" << std::endl;
      std::cerr <<  "what:" << what << std::endl;
      return false;
    }
    CNR_INFO(logger, "Controller initialized");

    // get joint name from the parameter server
    if (!nh.getParam("joints", my_joints_)){
      CNR_ERROR( logger, "Could not find joint name");
      return false;
    }
    if (my_joints_.size()!=6){
      CNR_ERROR( logger, "Wrong joint size");
      std::cout<<"Actual joint size:"<<my_joints_.size()<<std::endl;
      return false;
    }

    //Init handles
    for (size_t i=0;i<N_JOINTS;i++) {
      joints_.push_back(robot->getHandle(my_joints_.at(i)));  // throws on failure
    }
    if (joints_.size()!=N_JOINTS){
      CNR_ERROR( logger, "Wrong joint handle size");
      std::cout<<"Actual joint handle size:"<<joints_.size()<<std::endl;
      return false;
    }

    //Load base and tool frames
    std::string base_frame;
    std::string tool_frame;
    if (!nh.getParam("base_frame", base_frame)){
      CNR_ERROR( logger, "Could not find the base_frame parameters values");
      return false;
    }
    if (!nh.getParam("tool_frame", tool_frame)){
      CNR_ERROR( logger, "Could not find the tool_frame parameters values");
      return false;
    }

    //load robot model and parameters
    urdf::Model model;
    model.initParam("/robot_description");
    Eigen::Vector3d grav;
    grav << 0, 0, -9.806;

    chain_ = rosdyn::createChain(model, base_frame, tool_frame, grav);
    CNR_WARN(logger, "getActiveJointsNumber: " << chain_->getActiveJointsNumber());
    jacobian_.resize(6, chain_->getActiveJointsNumber());
    q_.resize(chain_->getActiveJointsNumber()); Dq_.resize(chain_->getActiveJointsNumber());
    q_.setZero(); Dq_.setZero();
    starting_pos_.resize(N_JOINTS);
    joint_command_.resize(N_JOINTS); 
    joint_command_test_.resize(N_JOINTS);

    //Read required parameters from controller configuration file
    if (!nh.getParam("inertia", inertia_)){
      CNR_ERROR( logger, "Could not find the inertia parameters values");
      return false;
    }
    if (!nh.getParam("damping_factor", damping_factor_)){
      CNR_ERROR( logger, "Could not find the damping factor parameter");
      return false;
    }
    if (!nh.getParam("stiffness", stiffness_)){
      CNR_ERROR( logger, "Could not find the stiffness parameters values");
      return false;
    }
    if (!nh.getParam("smoothing_factor_vel", smoothing_factor_vel_)){
      CNR_ERROR( logger, "Could not find the smoothing factor vel parameter");
      return false;
    }
    if (!nh.getParam("smoothing_factor_vel_pred", smoothing_factor_vel_pred_)){
      CNR_ERROR( logger, "Could not find the smoothing factor vel pred parameter");
      return false;
    }
    if (!nh.getParam("deadband_force", deadband_force_)){
      CNR_ERROR( logger, "Could not find the deadband force inf parameter");
      return false;
    }
    if (!nh.getParam("move_ref", MOVE_REF_POS_)){
      CNR_ERROR( logger, "Could not find the move ref parameter");
      return false;
    }
    if (!nh.getParam("move_ref_axis", move_ref_axis_)){
      CNR_ERROR( logger, "Could not find the move ref axis parameter");
      return false;
    }
    if ((move_ref_axis_<0)||(move_ref_axis_>2)){
      CNR_ERROR(logger, "move ref axis must be a value between 0 and 2");
      return false;
    }
    if (!nh.getParam("sin_amplitude", sin_amplitude_)){
      CNR_ERROR( logger, "Could not find the sin amplitude vel parameter");
      return false;
    }
    if (!nh.getParam("sin_frequency", sin_frequency_)){
      CNR_ERROR( logger, "Could not find the sin amplitude vel parameter");
      return false;
    }
    if (!nh.getParam("/simulated", simulated_)){
      CNR_ERROR( logger, "Could not find the simulated parameter");
      return false;
    }
    if (!nh.getParam("/human_estimator_gain", h_gain_)){
      CNR_ERROR( logger, "Could not find the prediction_params parameter");
      return false;
    }
    if (!nh.getParam("trajectory_waiting_cycles", trajectory_waiting_cycles)){
      ROS_ERROR("Could not find the waiting cycles parameter");
      return false;
    }


    myInertia_ << inertia_[0],inertia_[1],inertia_[2],inertia_[3],inertia_[4],inertia_[5];
    myStiffness_ << stiffness_[0],stiffness_[1],stiffness_[2],stiffness_[3],stiffness_[4],stiffness_[5];
    Stiffness_.resize(6,6); Damping_.resize(6,6); Inertia_.resize(6,6);
    Motor_gains_.resize(6,6); Inertia_pinv_.resize(6,6);
    Stiffness_.setZero(); Damping_.setZero(); Inertia_.setZero(); Inertia_pinv_.setZero();
    Stiffness_ = myStiffness_.asDiagonal();
    Inertia_ = myInertia_.asDiagonal();
    ext_wrench_.setZero(); ext_wrench_0_.setZero();
    pred_force_.setZero(); pred_pos_.setZero();
    starting_position_.setZero();
    virtual_reference_.setZero();
    cart_position_.setZero();
    ref_position_.setZero();
    human_ref_error.setZero();

    for (int i=0; i<N_JOINTS; i++){
      myDamping_(i) = 2*damping_factor_*sqrt(myStiffness_(i)*myInertia_(i));
      Inertia_pinv_(i,i) = 1.0/myInertia_(i);
    }
    Damping_ = myDamping_.asDiagonal();

    // Inertia_pinv_ = Inertia_.inverse();

    //start ft sensor subscriber
    std::string external_wrench_topic = "/robotiq_ft_wrench";
    std::string sensor_frame = "robotiq_ft_frame_id";
    sub_wrench_=nh.subscribe(external_wrench_topic, 5, &MyPositionController::readWrenchCB, this);
    chain_bs_ = rosdyn::createChain(model, base_frame, sensor_frame, grav);
    ext_wrench_enabled_ = false;
    experiment_window_ = false;

    //other publishers
    std::string position_reference_topic = "/custom_position_controller/reference_position";
    std::string actual_position_topic = "/custom_position_controller/actual_position";
    std::string wrench_topic = "/custom_position_controller/human_force_filtered";
    std::string cart_command_topic = "/custom_position_controller/cart_command";
    std::string joint_command_topic = "/custom_position_controller/joint_command";
    std::string human_estimated_ref_topic = "/custom_position_controller/human_estimated_ref";
    reference_pub_=nh.advertise<geometry_msgs::PoseStamped>(position_reference_topic,0);
    position_pub_=nh.advertise<geometry_msgs::PoseStamped>(actual_position_topic,0);
    pos_err_pub_=nh.advertise<geometry_msgs::PoseStamped>("/custom_position_controller/position_error",0);
    wrench_pub_ = nh.advertise<geometry_msgs::WrenchStamped>("/custom_position_controller/wrench", 0);
    cart_vel_filt_pub_=nh.advertise<geometry_msgs::PoseStamped>("/custom_position_controller/cart_vel_filtered",0);
    acc_command_pub_=nh.advertise<geometry_msgs::PoseStamped>("/custom_position_controller/acceleration",0);
    cart_command_pub_=nh.advertise<geometry_msgs::PoseStamped>(cart_command_topic,0);
    pred_vel_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/custom_position_controller/human_estimated_vel", 0);
    virtual_ref_enable_pub_=nh.advertise<std_msgs::Bool>("/custom_position_controller/virt_ref_enable",0);
    enable_hw_interface_pub_=nh.advertise<std_msgs::Bool>("/comau/enable_hw_interface_pub",0);
    h_ref_pred_sub_ = nh.subscribe(human_estimated_ref_topic, 1, &MyPositionController::readPredCB, this);
    exp_window_sub_ = nh.subscribe("/custom_position_controller/experiment_window", 1, &MyPositionController::expwinCB, this);
    virtual_ref_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/custom_position_controller/virtual_ref_position", 0);
    // loopxsec_pub_=nh.advertise<std_msgs::Float64>("/custom_position_controller/loop_per_sec",0);

    FIRST_LOOP_=false;
    enable_hw_msg_.data=false;
    enable_hw_interface_pub_.publish(enable_hw_msg_);
    CNR_RETURN_TRUE(logger);
    }


    void starting(const ros::Time &time) override  {
      CNR_INFO( logger, "starting custom postition controller");
      CNR_TRACE_START_THROTTLE_DEFAULT(logger);
      for (size_t i=0;i<N_JOINTS;i++) {
        starting_pos_(i)=joints_.at(i).getPosition(); //save position at starting
        joint_command_(i)=joints_.at(i).getPosition(); //set goal to current position
      }

      q_<< starting_pos_[0], starting_pos_[1], starting_pos_[2], starting_pos_[3], starting_pos_[4], starting_pos_[5];
      jacobian_=chain_->getJacobian(q_); //save jacobian at starting

      starting_pose_ = chain_->getTransformation(starting_pos_);
      starting_position_ = starting_pose_.translation();
      virtual_reference_ = starting_position_;
      cart_position_= starting_position_;
      ref_position_= starting_position_;
      wrench_of_tool_in_base_.setZero();
      pred_force_in_base_.setZero();
      cart_command_ = starting_position_;
      test_command_ = starting_position_;
      pred_pos_ = starting_position_;

      FIRST_LOOP_=true;
      enable_hw_msg_.data=false;
      enable_hw_interface_pub_.publish(enable_hw_msg_);
      CNR_RETURN_OK_THROTTLE_DEFAULT(logger, void());
    }


    void stopping(const ros::Time &time) override  {
      CNR_INFO( logger, "Bye.");
    }


    void update(const ros::Time &time, const ros::Duration &period) override
    {
        // CNR_INFO( logger, "UPDATE FUNCTION");
        CNR_TRACE_START_THROTTLE_DEFAULT(logger);
        ros::Duration loop_t_chrono(1);
        geometry_msgs::PoseStamped ref_msg, actual_position_msg, cart_command_msg, pred_vel_msg_, acc_msg, vel_filt_msg, vel_msg, pos_err_msg, virtual_ref_msg;
        geometry_msgs::WrenchStamped wrench_msg;
        std_msgs::Bool virtual_ref_enable_msg;
        std_msgs::Float64 loopxsec_msg;
        Eigen::Vector6d seed;
        double loop_per_sec, traj_end_time;
        double rand_x, rand_y;
        static int counter=0;

        if(FIRST_LOOP_){
          loop_cycles_counter_ = 0;
          cart_position_prev_= cart_position_; 
          cart_command_vel_.setZero();
          pos_err_.setZero(); pos_dot_.setZero(); pos_err_Ddot_.setZero();
          pos_dot_filtered_.setZero();
          joint_command_.setZero(); joint_command_test_.setZero();
          cart_command_transf_.setIdentity(); cart_command_test_transf_.setIdentity();
          loop_per_sec = 0;
          loop_time_ = 0;
          t_start_ = 0;
          traj_end_time = 0.0;
          rand_x = rand_y = t_0 = 0;
          MOVE  = false;
          virt_ref_enable_ = false;
          pred_time_ = pred_exec_time_ = 0.0;
          pred_vel_x_ = pred_vel_y_ = 0.0;
          pred_vel_filt_x_ = pred_vel_filt_y_ = 0.0;
        }
        enable_hw_msg_.data = true;
        enable_hw_interface_pub_.publish(enable_hw_msg_);
            
        // CNR_INFO( logger, "Compute control loop time and frequency");
        loop_time_ += period.toSec();
        loop_cycles_counter_ ++;
        // loop_per_sec = loop_cycles_counter_/loop_time_;

        // CNR_INFO( logger, "Update position, velocity and joint torque");
        for (size_t i=0;i<N_JOINTS;i++) {
          q_[i]=joints_.at(i).getPosition();
        }

        // CNR_INFO( logger, "Cartesian Transformations");
        cart_pose_ = chain_->getTransformation(q_);
        cart_position_ = cart_pose_.translation();
        cart_command_transf_ = chain_->getTransformation(q_);
        cart_command_test_transf_ = chain_->getTransformation(q_);

        for(int idx=0; idx<3; idx++){
          if(abs(cart_position_[idx] - cart_position_prev_[idx]) < 0.0001) cart_position_[idx] = cart_position_prev_[idx];
        }

        //SET NEW VIRTUAL REFERENCE
        std::random_device rd;
        std::default_random_engine eng(rd());
        std::uniform_real_distribution<> distr(-0.1, 0.1);
        if(loop_cycles_counter_>trajectory_waiting_cycles){
          double t, holding_time;
          if(loop_cycles_counter_==trajectory_waiting_cycles+1){
            t_0 = time.now().toSec();
            holding_time=0.0;
            virt_ref_enable_ = true;
          }
      
          if((std::abs(cart_position_[0]-virtual_reference_[0])<0.05)&&(std::abs(cart_position_[1]-virtual_reference_[1])<0.05)){
            t = time.now().toSec();
            holding_time = t-t_0;
            if(holding_time>3.0){
              rand_x = distr(eng);
              rand_y = distr(eng);
              virtual_reference_[0] =  starting_position_[0] + (++counter)%2*(rand_x);
              virtual_reference_[1] =  starting_position_[1] + counter%2*(rand_y);
              //           std::cout<<"rand_x: "<<(rand_x/100)<<"\t"<<"rand_y: "<<(rand_y/100)<<"\t"<<"rand_z: "<<(rand_z/100)<<std::endl;
            }
          }
          else {
            t_0 = time.now().toSec();
            holding_time=0.0;
          }
        }

        // CNR_INFO(logger, "Set Reference position")
        if(MOVE_REF_POS_ && experiment_window_){

          // pred_exec_time_ += period.toSec();
          // double time_diff = pred_time_ - time.now().toNSec();
          // std::cout << "Time_diff custom pos: " << time_diff << std::endl;
          pred_vel_x_ = (1/h_gain_) * pred_force_in_base_[0];
          pred_vel_y_ = (1/h_gain_) * pred_force_in_base_[1];

          pred_vel_filt_x_ = smoothing_factor_vel_pred_*pred_vel_x_ + (1-smoothing_factor_vel_pred_)*pred_vel_filt_x_;
          pred_vel_filt_y_ = smoothing_factor_vel_pred_*pred_vel_y_ + (1-smoothing_factor_vel_pred_)*pred_vel_filt_y_;

          pred_pos_[0] = pred_pos_[0] + pred_vel_filt_x_ * period.toSec();
          pred_pos_[1] = pred_pos_[1] + pred_vel_filt_y_ * period.toSec();
          

          if ((abs(pred_pos_(0) - ref_position_(0))>0.005)&&(abs(pred_pos_(0) - ref_position_(0)<0.5))&&(abs(pred_pos_(1) - ref_position_(1))>0.005)&&(abs(pred_pos_(1) - ref_position_(1)<0.5))&&(MOVE==false)){
            t_start_ = time.now().toSec();
            ref_position_[0] = pred_pos_[0];
            ref_position_[1] = pred_pos_[1];
            MOVE = true;
          }
          // else{
          //   ref_position_[0] = starting_position_[0];
          //   ref_position_[1] = starting_position_[1];
          // }
          if(MOVE==true){
            // traj_end_time = t_start_ + pred_exec_time_; 
            // ref_position_(0) = (pred_pos_[0] - starting_position_[0])*((loop_time_ - t_start_)/(traj_end_time - t_start_));
            // ref_position_(1) = (pred_pos_[1] - starting_position_[1])*((loop_time_ - t_start_)/(traj_end_time - t_start_));
            ref_position_[0] = pred_pos_[0];
            ref_position_[1] = pred_pos_[1];
          }
          if ((abs(pred_pos_(0) - ref_position_(0))<0.001)&&(abs(pred_pos_(1) - ref_position_(1))<0.001)&&(MOVE == true)){
            ref_position_ = starting_position_ = cart_command_;
            t_start_ = loop_time_;
            pred_exec_time_ = 0.0;
            MOVE = false;
            // CNR_INFO(logger, "RESET");
          }

          // ref_position_(move_ref_axis_) = starting_position_(move_ref_axis_) + sin_amplitude_*sin(2*M_PI*sin_frequency_*loop_time_);
    
        }

        // CNR_INFO( logger, "Impedance controller");
        // human_ref_error = virtual_reference_ - cart_position_;
        for(int i=0; i<3; i++){
          pos_err_[i] =  cart_command_[i] - ref_position_[i];
          pos_dot_[i] = cart_command_vel_[i]; // - cart_command_vel_prev[i];
        }
          
        pos_dot_filtered_ = smoothing_factor_vel_*pos_dot_ + (1-smoothing_factor_vel_)*pos_dot_filtered_;

        for(int i=0; i<6; i++){
          if(abs(wrench_of_tool_in_base_(i))<deadband_force_) wrench_of_tool_in_base_(i)=0;
        }

        pos_err_Ddot_ = Inertia_pinv_*(wrench_of_tool_in_base_ - Damping_*pos_dot_filtered_ - Stiffness_*pos_err_);

        // CNR_INFO( logger, "Send Command");
        cart_command_vel_[0] = cart_command_vel_[0] + pos_err_Ddot_[0]*period.toSec();
        cart_command_vel_[1] = cart_command_vel_[1] + pos_err_Ddot_[1]*period.toSec();
        cart_command_vel_[2] = cart_command_vel_[2] + pos_err_Ddot_[2]*period.toSec();

        cart_command_[0] = cart_command_[0] + cart_command_vel_[0]*period.toSec();
        cart_command_[1] = cart_command_[1] + cart_command_vel_[1]*period.toSec();
        cart_command_[2] = cart_command_[2] + cart_command_vel_[2]*period.toSec();

        //Generate test sin trajectory without impedance control
        // cart_command_[0] = starting_position_[0] + sin_amplitude_*sin(2*M_PI*(sin_frequency_/2)*loop_time_);
        // cart_command_[1] = starting_position_[1] + sin_amplitude_*sin(2*M_PI*sin_frequency_*loop_time_);
        // cart_command_ = starting_position_;

        seed = q_; 
        cart_command_transf_.translation() = cart_command_;
        bool IK_success = chain_->computeLocalIk(joint_command_, cart_command_transf_, seed);

        // cart_command_test_transf_.translation() = test_command_;
        // chain_->computeLocalIk(joint_command_test_, cart_command_test_transf_, seed);

        if(IK_success){
          for (size_t i=0;i<N_JOINTS;i++) {
            joints_.at(i).setCommand(joint_command_(i));
          }
        }
        else{
          CNR_ERROR( logger, "FAILED TO COMPUTE INVERSE KINEMATICS!!");
        }

        // CNR_INFO( logger, "PUBLISHERS");
        ref_msg.pose.position.x = ref_position_.x();
        ref_msg.pose.position.y = ref_position_.y();
        ref_msg.pose.position.z = ref_position_.z();
        ref_msg.header.stamp = time.now();
        ref_msg.header.frame_id = "base_link";
        reference_pub_.publish(ref_msg);
        actual_position_msg.pose.position.x = cart_position_.x();
        actual_position_msg.pose.position.y = cart_position_.y();
        actual_position_msg.pose.position.z = cart_position_.z();
        actual_position_msg.header.stamp = time.now();
        actual_position_msg.header.frame_id = "base_link";
        position_pub_.publish(actual_position_msg);
        wrench_msg.wrench.force.x = wrench_of_tool_in_base_.x();
        wrench_msg.wrench.force.y = wrench_of_tool_in_base_.y();
        wrench_msg.wrench.force.z = wrench_of_tool_in_base_.z();
        wrench_msg.header.stamp = time.now();
        wrench_msg.header.frame_id = "base_link";
        wrench_pub_.publish(wrench_msg);
        vel_filt_msg.pose.position.x = pos_dot_filtered_.x();
        vel_filt_msg.pose.position.y = pos_dot_filtered_.y();
        vel_filt_msg.pose.position.z = pos_dot_filtered_.z();
        vel_filt_msg.header.stamp = time.now();
        vel_filt_msg.header.frame_id = "base_link";
        cart_vel_filt_pub_.publish(vel_filt_msg);
        pos_err_msg.pose.position.x = pos_err_.x();
        pos_err_msg.pose.position.y = pos_err_.y();
        pos_err_msg.pose.position.z = pos_err_.z();
        pos_err_msg.header.stamp = time.now();
        pos_err_msg.header.frame_id = "base_link";
        pos_err_pub_.publish(pos_err_msg);
        acc_msg.pose.position.x = pos_err_Ddot_.x();
        acc_msg.pose.position.y = pos_err_Ddot_.y();
        acc_msg.pose.position.z = pos_err_Ddot_.z();
        acc_msg.header.stamp = time.now();
        acc_msg.header.frame_id = "base_link";
        acc_command_pub_.publish(acc_msg);
        virtual_ref_msg.pose.position.x = virtual_reference_.x();
        virtual_ref_msg.pose.position.y = virtual_reference_.y();
        virtual_ref_msg.pose.position.z = ref_position_.z();
        virtual_ref_msg.header.stamp = time.now();
        virtual_ref_msg.header.frame_id = "base_link";
        virtual_ref_pub_.publish(virtual_ref_msg);
        pred_vel_msg_.pose.position.x = pred_vel_filt_x_;
        pred_vel_msg_.pose.position.y = pred_vel_filt_y_;
        pred_vel_msg_.pose.position.z = pos_dot_filtered_.z();
        pred_vel_msg_.header.stamp = time.now();
        pred_vel_msg_.header.frame_id = "base_link";
        pred_vel_pub_.publish(pred_vel_msg_);
        cart_command_msg.pose.position.x = cart_command_.x();
        cart_command_msg.pose.position.y = cart_command_.y();
        cart_command_msg.pose.position.z = cart_command_.z();
        cart_command_msg.header.stamp = time.now();
        cart_command_msg.header.frame_id = "base_link";
        car2t_command_pub_.publish(cart_command_msg);
        virtual_ref_enable_msg.data = virt_ref_enable_;
        virtual_ref_enable_pub_.publish(virtual_ref_enable_msg);
        // loopxsec_msg.data = loop_per_sec;
        // loopxsec_pub_.publish(loopxsec_msg);

        cart_position_prev_ = cart_position_;
        FIRST_LOOP_=false;
        
        CNR_RETURN_OK_THROTTLE_DEFAULT(logger, void());
    }


    void readWrenchCB(const geometry_msgs::WrenchStamped& msg)
    {
      if(!ext_wrench_enabled_){
        ext_wrench_0_(0)=msg.wrench.force.x;
        ext_wrench_0_(1)=msg.wrench.force.y;
        ext_wrench_0_(2)=msg.wrench.force.z;
        ext_wrench_0_(3)=msg.wrench.torque.x;
        ext_wrench_0_(4)=msg.wrench.torque.y;
        ext_wrench_0_(5)=msg.wrench.torque.z;

        ext_wrench_enabled_ = true;
      }

      ext_wrench_(0)=msg.wrench.force.x - ext_wrench_0_(0);
      ext_wrench_(1)=msg.wrench.force.y - ext_wrench_0_(1);
      ext_wrench_(2)=msg.wrench.force.z - ext_wrench_0_(2);
      ext_wrench_(3)=msg.wrench.torque.x - ext_wrench_0_(3);
      ext_wrench_(4)=msg.wrench.torque.y - ext_wrench_0_(4);
      ext_wrench_(5)=msg.wrench.torque.z - ext_wrench_0_(5);

      Eigen::Affine3d T_base_tool=chain_->getTransformation(q_);
      // Eigen::MatrixXd jacobian_of_tool_in_base = chain_->getJacobian(q_);
      Eigen::Affine3d T_base_sensor=chain_bs_->getTransformation(q_);
      Eigen::Affine3d T_tool_sensor= T_base_tool.inverse()*T_base_sensor;

      Eigen::Vector6d wrench_of_tool_in_tool = rosdyn::spatialDualTranformation (ext_wrench_ , T_tool_sensor         );
      wrench_of_tool_in_base_ = rosdyn::spatialRotation          (wrench_of_tool_in_tool     , T_base_tool.linear()  );
    }

    void readPredCB(const geometry_msgs::WrenchStamped& msg){
      pred_force_(0) = msg.wrench.force.x;
      pred_force_(1) = msg.wrench.force.y;
      pred_force_(2)= 0.0;
      pred_force_(3)= 0.0;
      pred_force_(4)= 0.0;
      pred_force_(5)= 0.0;

      Eigen::Affine3d T_base_tool_2=chain_->getTransformation(q_);
      Eigen::Affine3d T_base_sensor_2=chain_bs_->getTransformation(q_);
      Eigen::Affine3d T_tool_sensor_2= T_base_tool_2.inverse()*T_base_sensor_2;

      Eigen::Vector6d wrench_of_tool_in_tool_2 = rosdyn::spatialDualTranformation (pred_force_ , T_tool_sensor_2         );
      pred_force_in_base_ = rosdyn::spatialRotation          (wrench_of_tool_in_tool_2     , T_base_tool_2.linear()  );
      ros::Time time_rcv = msg.header.stamp;
      pred_time_ = time_rcv.toNSec();
    }

    void expwinCB(const std_msgs::Bool& msg){
      experiment_window_ = msg.data;
    }



    private:
      mutable cnr_logger::TraceLogger logger;
      std::vector<hardware_interface::JointHandle> joints_;
      std::string myTCP;
      shared_ptr_namespace::shared_ptr<rosdyn::Chain> chain_, chain_bs_;
      ros::Publisher reference_pub_, position_pub_, cart_command_pub_, virtual_ref_enable_pub_, cart_force_filt_pub_, wrench_pub_, virtual_ref_pub_, enable_hw_interface_pub_, loopxsec_pub_, cart_vel_filt_pub_, vel_pub_, acc_command_pub_, pred_vel_pub_, pos_err_pub_;
      ros::Subscriber sub_wrench_, h_ref_pred_sub_, exp_window_sub_;
      std_msgs::Bool enable_hw_msg_;
      
      Eigen::VectorXd joint_command_, joint_command_test_, starting_pos_;
      Eigen::Vector6d myStiffness_, myDamping_, myInertia_, myGains_, ext_wrench_, ext_wrench_0_;
      Eigen::VectorXd q_, Dq_;
      Eigen::MatrixXd jacobian_, Inertia_, Stiffness_, Damping_, Motor_gains_, Inertia_pinv_;
      std::vector<double> inertia_, stiffness_;
      double pred_params_[2], h_gain_, t_0;
      bool ext_wrench_enabled_, experiment_window_, virt_ref_enable_;
      Eigen::Affine3d starting_pose_;

      //Control Loop variables
      bool FIRST_LOOP_, MOVE_REF_POS_, simulated_, MOVE;
      int loop_cycles_counter_, trajectory_waiting_cycles, move_ref_axis_;
      double loop_time_, t_start_, pred_time_, pred_exec_time_, pred_vel_x_, pred_vel_y_, pred_vel_filt_x_, pred_vel_filt_y_;
      double smoothing_factor_vel_, smoothing_factor_vel_pred_, deadband_force_, damping_factor_;
      Eigen::Vector3d starting_position_, virtual_reference_, ref_position_, pred_pos_, cart_position_, cart_position_prev_, human_ref_error, test_command_, cart_command_, cart_command_vel_;
      Eigen::Vector6d pos_err_, pos_dot_, pos_dot_filtered_, pos_err_Ddot_, pred_force_;
      Eigen::Affine3d cart_pose_, cart_command_transf_, cart_command_test_transf_;
      Eigen::Vector6d wrench_of_tool_in_base_, pred_force_in_base_;
      double sin_amplitude_, sin_frequency_;
      std::vector<std::string> my_joints_;
      // ros::Time pred_time_;
      std::mutex mtx;
  };
  
  PLUGINLIB_EXPORT_CLASS(custom_position_controller::MyPositionController, controller_interface::ControllerBase);
}
