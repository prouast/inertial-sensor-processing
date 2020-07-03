from math import sqrt, atan2, asin, degrees, radians
import quaternion
import numpy as np

class MadgwickFusion:
  """6 DOF fusion with acclerometer and gyroscope."""
  def __init__(self, q, dt):
    self.q = q
    self.dt = dt
    self.beta = 0.25

  def update(self, acc, gyro):
    """One step on Madgwick's algorithm.

    Arguments
      acc: Accelerometer reading. Unit not important since it is normalized
      gyro: Gyroscope reading [degrees per second].
    """
    a = np.quaternion(0, acc[0], acc[1], acc[2]).normalized()
    g = np.quaternion(0, radians(gyro[0]), radians(gyro[1]), radians(gyro[2]))
    q = self.q

    # Temp variables
    _2qw, _2qx, _2qy, _2qz = 2 * q.w, 2 * q.x, 2 * q.y, 2 * q.z
    _4qw, _4qx, _4qy = 4 * q.w, 4 * q.x, 4 * q.y
    _8qx, _8qy  = 8 * q.x, 8 * q.y
    qwqw, qxqx, qyqy, qzqz = q.w * q.w, q.x * q.x, q.y * q.y, q.z * q.z

    # Gradient descent step
    s = np.quaternion(
      _4qw * qyqy + _2qy * a.x + _4qw * qxqx - _2qx * a.y,
      _4qx * qzqz - _2qz * a.x + 4 * qwqw * q.x - _2qw * a.y - _4qx + _8qx * qxqx + _8qx * qyqy + _4qx * a.z,
      4 * qwqw * q.y + _2qw * a.x + _4qy * qzqz - _2qz * a.y - _4qy + _8qy * qxqx + _8qy * qyqy + _4qy * a.z,
      4 * qxqx * q.z - _2qx * a.x + 4 * qyqy * q.z - _2qy * a.y)
    s = s.normalized()

    # Compute quaternion rate of change
    qdot = (q * g) * 0.5 - self.beta * s

    # Integrate
    q += qdot * 1 / self.dt
    self.q = q.normalized()
