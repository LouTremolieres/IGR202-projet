#version 450 core            // minimal GL version support expected from the GPU

struct LightSource {
  vec3 position;
  vec3 color;
  float intensity;
  int isActive;
};

int numberOfLights = 3;
uniform LightSource lightSources[3];
// TODO: shadow maps
uniform int isTexturedNormal;
uniform int isTexturedColor;

struct Material {
  vec3 albedo;
  sampler2D normalTex;
  sampler2D albedoTex;
  // TODO: textures
};

uniform Material material;

uniform vec3 camPos;

in vec3 fPositionModel;
in vec3 fPosition;
in vec3 fNormal;
in vec2 fTexCoord;

out vec4 colorOut; // shader output: the color response attached to this fragment
uniform mat4 modelMat, viewMat, projMat;


in float isContour;
in float isSuggestiveContour;

float pi = 3.1415927;

// TODO: shadows
void main() {
  
  vec3 n;
  if(isTexturedNormal != 0){
    n = normalize(2*texture(material.normalTex, fTexCoord).rgb-1);
  }
  else {
    n = normalize(fNormal);
  }

  vec3 wo = normalize(camPos - fPosition); // unit vector pointing to the camera

  vec3 radiance = vec3(0, 0, 0);
  for(int i=0; i<numberOfLights; ++i) {
    LightSource a_light = lightSources[i];
    if(a_light.isActive == 1) { // consider active lights only
      vec3 wi = normalize(a_light.position - fPosition); // unit vector pointing to the light
      vec3 Li = a_light.color*a_light.intensity;
      
      vec3 albedo;
      if(isTexturedColor != 0){
        albedo = texture(material.albedoTex, fTexCoord).rgb;
      }
      else {
        albedo = material.albedo;
      }
      //vec3 albedo = texture(material.normalTex, fTexCoord).rgb;


      radiance += Li*albedo*max(dot(n, wi), 0);
    }
  }

  //colorOut = vec4(radiance, 1.0); // build an RGBA value from an RGB one
 
  colorOut = vec4(0.);//vec4(material.albedo, 1.0);
  //colorOut += isContour * vec4(1.0, 0.0, 0.0, 0.0); 
  colorOut += isSuggestiveContour * vec4(0.0, 0.0, 1.0, 0.0);
  
}
