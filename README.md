# Bernard: Bipedal Experimental RL-based Neuromorphic Autonomous Reconfigurable Droid

## Overview

This repository contains all the necessary components to build and operate the BERNARD bipedal robot. It includes submodules for the robot's firmware and reinforcement learning environment, as well as 3D models for printing, a Bill of Materials (BOM), and purchase-related information.

The project is developed as part of a master's thesis at AGH University of Krakow.

---

## Submodules

### [STM32 ROS Bernard Node](https://github.com/Baey/bernard-stm32-ros-node)

This submodule contains the **PlatformIO project** for the micro-ROS node, which handles communication, sensor data reading, and GUI display. Key features include:
- **Micro-ROS Integration**: Real-time communication between the STM32 microcontroller and ROS 2.
- **Sensor Management**: Reads data from IMU and foot pressure sensors.
- **GUI**: Displays system status on a TFT screen.

### [Bernard RL](https://github.com/Baey/bernard-rl)

This submodule provides an IsaacLab extension with a reinforcement learning environment for training the robot in simulation. Key features include:
- **Isolation**: Work outside the core Isaac Lab repository.
- **Flexibility**: Run as an extension in Omniverse.

---

## 3D Models

The repository includes 3D models for printing the robot's parts. These models are located in the `3d_models/` directory. Ensure you use the recommended materials and settings for optimal results.

---

## Bill of Materials (BOM)

A detailed BOM is provided in the `BOM/` directory. It includes:
- Part names and descriptions.
- Quantities required.
- Recommended suppliers and links for purchase.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone --recurse-submodules https://github.com/Baey/bernard-bipedal-robot.git
   ```

2. **Initialize Submodules**:
   ```bash
   cd bernard-bipedal-robot
   git submodule update --init --recursive
   ```

3. Follow the installation instructions in the respective submodules for firmware and RL environment setup.

---

## Acknowledgments

This project is part of a master's thesis at AGH University of Krakow.