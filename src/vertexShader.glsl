#version 450 core            // minimal GL version support expected from the GPU

layout(location=0) in vec3 vPosition; // the 1st input attribute is the position (CPU side: glVertexAttrib 0)
layout(location=1) in vec3 vNormal;
layout(location=2) in vec2 vTexCoord;
layout(location=3) in float vIsSuggContour;


uniform mat4 modelMat, viewMat, projMat;
uniform mat3 normMat;

out vec3 fPositionModel;
out vec3 fPosition;
out vec3 fNormal;
out vec2 fTexCoord;
out float isContour;
out float isSuggestiveContour;

void main() {
  
  fPositionModel = vPosition;
  fPosition = (modelMat*vec4(vPosition, 1.0)).xyz;
  fNormal = normMat*vNormal;
  fTexCoord = vTexCoord;

  gl_Position =  projMat*viewMat*modelMat*vec4(vPosition, 1.0); // mandatory

  vec3 camPos = inverse(viewMat)[3].xyz;
  vec3 v = camPos - fPosition;
  if(dot(v,fNormal) <= 0.0005) {
    isContour = 1;
  }
  else {isContour = 0;}

  isSuggestiveContour = vIsSuggContour;
}
