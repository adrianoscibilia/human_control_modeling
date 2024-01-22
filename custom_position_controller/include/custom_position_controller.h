#ifndef CONTROLLER_ADRIANO_H
#define CONTROLLER_ADRIANO_H

#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>

namespace controller_adriano{

class MyPositionController : public controller_interface::Controller<hardware_interface::PositionJointInterface>
{
public:

  bool init(hardware_interface::PositionJointInterface* hw, ros::NodeHandle &n) override;

  void update(const ros::Time& time, const ros::Duration& period) override;

  void starting(const ros::Time& time) override;

  void stopping(const ros::Time& time) override;

  void setCommandCB(const std_msgs::Float64ConstPtr& msg);

private:
  std::vector<hardware_interface::JointHandle> joints_;
  double gain_;
  double command_[6];
  ros::Subscriber sub_command_;
  ros::Publisher current_command_;
};

}//namespace

#endif // CONTROLLER_ADRIANO_H
