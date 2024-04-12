#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT    ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict iimage3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION restrict           Block {
  ivec4 size;
  vec2 clamp;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.size.xyz))) {
    vec4 temp = imageLoad(uOutput, pos);
    temp = clamp(temp, uBlock.clamp.x, uBlock.clamp.y);
    ivec4 store = ivec4(temp);
    imageStore(
        uOutput,
        pos,
        store);
  }
}
