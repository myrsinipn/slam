import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <worldbody>
    <body pos="0 0 0">
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)