#define _USE_MATH_DEFINES

#include "Mesh.h"

// --- PROJET ---
#include "Camera.h"
// --------------

#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <exception>
#include <ios>
#include <string>
#include <memory>

Mesh::~Mesh()
{
  clear();
}

void Mesh::computeBoundingSphere(glm::vec3 &center, float &radius) const
{
  center = glm::vec3(0.0);
  radius = 0.f;
  for(const auto &p : _vertexPositions)
    center += p;
  center /= _vertexPositions.size();
  for(const auto &p : _vertexPositions)
    radius = std::max(radius, distance(center, p));
}

void Mesh::recomputePerVertexNormals(bool angleBased)
{
  _vertexNormals.clear();
  // Change the following code to compute a proper per-vertex normal
  _vertexNormals.resize(_vertexPositions.size(), glm::vec3(0.0, 0.0, 0.0));

  for(unsigned int tIt=0 ; tIt < _triangleIndices.size() ; ++tIt) {
    glm::uvec3 t = _triangleIndices[tIt];
    glm::vec3 n_t = glm::cross(
      _vertexPositions[t[1]] - _vertexPositions[t[0]],
      _vertexPositions[t[2]] - _vertexPositions[t[0]]);
    _vertexNormals[t[0]] += n_t;
    _vertexNormals[t[1]] += n_t;
    _vertexNormals[t[2]] += n_t;
  }
  for(unsigned int nIt = 0 ; nIt < _vertexNormals.size() ; ++nIt) {
    glm::normalize(_vertexNormals[nIt]);
  }
}

void Mesh::recomputePerVertexTextureCoordinates()
{
  _vertexTexCoords.clear();
  // Change the following code to compute a proper per-vertex texture coordinates
  _vertexTexCoords.resize(_vertexPositions.size(), glm::vec2(0.0, 0.0));

  float xMin = FLT_MAX, xMax = FLT_MIN;
  float yMin = FLT_MAX, yMax = FLT_MIN;
  for(glm::vec3 &p : _vertexPositions) {
    xMin = std::min(xMin, p[0]);
    xMax = std::max(xMax, p[0]);
    yMin = std::min(yMin, p[1]);
    yMax = std::max(yMax, p[1]);
  }
  for(unsigned int pIt = 0 ; pIt < _vertexTexCoords.size() ; ++pIt) {
    _vertexTexCoords[pIt] = glm::vec2(
      (_vertexPositions[pIt][0] - xMin)/(xMax-xMin),
      (_vertexPositions[pIt][1] - yMin)/(yMax-yMin));
  }
}

void Mesh::addPlan(float square_half_side)
{
  _vertexPositions.push_back(glm::vec3(-square_half_side,-square_half_side, 0));
  _vertexPositions.push_back(glm::vec3(+square_half_side,-square_half_side, 0));
  _vertexPositions.push_back(glm::vec3(+square_half_side,+square_half_side, 0));
  _vertexPositions.push_back(glm::vec3(-square_half_side,+square_half_side, 0));

  _vertexTexCoords.push_back(glm::vec2(0.0, 0.0));
  _vertexTexCoords.push_back(glm::vec2(1.0, 0.0));
  _vertexTexCoords.push_back(glm::vec2(1.0, 1.0));
  _vertexTexCoords.push_back(glm::vec2(0.0, 1.0));

  _vertexNormals.push_back(glm::vec3(0,0, 1));
  _vertexNormals.push_back(glm::vec3(0,0, 1));
  _vertexNormals.push_back(glm::vec3(0,0, 1));
  _vertexNormals.push_back(glm::vec3(0,0, 1));

  _triangleIndices.push_back(
    glm::uvec3(_vertexPositions.size()-4, _vertexPositions.size()-3, _vertexPositions.size()-2));
  _triangleIndices.push_back(
    glm::uvec3(_vertexPositions.size()-4, _vertexPositions.size()-2, _vertexPositions.size()-1));
}

void Mesh::init(glm::vec3 camPos)
{
  computeSuggContours(camPos);

  glCreateBuffers(1, &_posVbo); // Generate a GPU buffer to store the positions of the vertices
  size_t vertexBufferSize = sizeof(glm::vec3)*_vertexPositions.size(); // Gather the size of the buffer from the CPU-side vector
  glNamedBufferStorage(_posVbo, vertexBufferSize, _vertexPositions.data(), GL_DYNAMIC_STORAGE_BIT); // Create a data store on the GPU

  glCreateBuffers(1, &_normalVbo); // Same for normal
  glNamedBufferStorage(_normalVbo, vertexBufferSize, _vertexNormals.data(), GL_DYNAMIC_STORAGE_BIT);

  glCreateBuffers(1, &_texCoordVbo); // Same for texture coordinates
  size_t texCoordBufferSize = sizeof(glm::vec2)*_vertexTexCoords.size();
  glNamedBufferStorage(_texCoordVbo, texCoordBufferSize, _vertexTexCoords.data(), GL_DYNAMIC_STORAGE_BIT);

  glCreateBuffers(1, &_ibo); // Same for the index buffer, that stores the list of indices of the triangles forming the mesh
  size_t indexBufferSize = sizeof(glm::uvec3)*_triangleIndices.size();
  glNamedBufferStorage(_ibo, indexBufferSize, _triangleIndices.data(), GL_DYNAMIC_STORAGE_BIT);

  glCreateBuffers(1, &_isSuggContourVbo); // Same for if it is a suggestive contour
  size_t isSuggContourBufferSize = sizeof(glm::vec2)*_isSuggContour.size();
  glNamedBufferStorage(_isSuggContourVbo, isSuggContourBufferSize, _isSuggContour.data(), GL_DYNAMIC_STORAGE_BIT);

  glCreateVertexArrays(1, &_vao); // Create a single handle that joins together attributes (vertex positions, normals) and connectivity (triangles indices)
  glBindVertexArray(_vao);

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, _posVbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), 0);

  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, _normalVbo);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), 0);

  glEnableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, _texCoordVbo);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);

  glEnableVertexAttribArray(3);
  glBindBuffer(GL_ARRAY_BUFFER, _isSuggContourVbo);
  glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
  glBindVertexArray(0); // Desactive the VAO just created. Will be activated at rendering time.
}

void Mesh::initOldGL()
{
  // Generate a GPU buffer to store the positions of the vertices
  size_t vertexBufferSize = sizeof(glm::vec3)*_vertexPositions.size();
  glGenBuffers(1, &_posVbo);
  glBindBuffer(GL_ARRAY_BUFFER, _posVbo);
  glBufferData(GL_ARRAY_BUFFER, vertexBufferSize, _vertexPositions.data(), GL_DYNAMIC_READ);

  // Same for normal
  glGenBuffers(1, &_normalVbo);
  glBindBuffer(GL_ARRAY_BUFFER, _normalVbo);
  glBufferData(GL_ARRAY_BUFFER, vertexBufferSize, _vertexNormals.data(), GL_DYNAMIC_READ);

  // Same for texture coordinates
  size_t texCoordBufferSize = sizeof(glm::vec2)*_vertexTexCoords.size();
  glGenBuffers(1, &_texCoordVbo);
  glBindBuffer(GL_ARRAY_BUFFER, _texCoordVbo);
  glBufferData(GL_ARRAY_BUFFER, texCoordBufferSize, _vertexTexCoords.data(), GL_DYNAMIC_READ);

  // Same for the index buffer that stores the list of indices of the triangles forming the mesh
  size_t indexBufferSize = sizeof(glm::uvec3)*_triangleIndices.size();
  glGenBuffers(1, &_ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBufferSize, _triangleIndices.data(), GL_DYNAMIC_READ);

  //Same for if it is a suggestive contour
  size_t isSuggContourBufferSize = sizeof(float)*_isSuggContour.size();
  glGenBuffers(1, &_isSuggContourVbo);
  glBindBuffer(GL_ARRAY_BUFFER, _isSuggContourVbo);
  glBufferData(GL_ARRAY_BUFFER, isSuggContourBufferSize, _isSuggContour.data(), GL_DYNAMIC_READ);

  // Create a single handle that joins together attributes (vertex positions, normals) and connectivity (triangles indices)
  glGenVertexArrays(1, &_vao);
  glBindVertexArray(_vao);

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, _posVbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), 0);

  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, _normalVbo);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), 0);

  glEnableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, _texCoordVbo);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), 0);

  glEnableVertexAttribArray(3);
  glBindBuffer(GL_ARRAY_BUFFER, _isSuggContourVbo);
  glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT), 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);

  glBindVertexArray(0); // Desactive the VAO just created. Will be activated at rendering time.
}

void Mesh::render()
{
  glBindVertexArray(_vao);      // Activate the VAO storing geometry data
  glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(_triangleIndices.size()*3), GL_UNSIGNED_INT, 0);
  // Call for rendering: stream the current GPU geometry through the current GPU program
}

void Mesh::clear()
{
  _vertexPositions.clear();
  _vertexNormals.clear();
  _vertexTexCoords.clear();
  _triangleIndices.clear();
  if(_vao) {
    glDeleteVertexArrays(1, &_vao);
    _vao = 0;
  }
  if(_posVbo) {
    glDeleteBuffers(1, &_posVbo);
    _posVbo = 0;
  }
  if(_normalVbo) {
    glDeleteBuffers(1, &_normalVbo);
    _normalVbo = 0;
  }
  if(_texCoordVbo) {
    glDeleteBuffers(1, &_texCoordVbo);
    _texCoordVbo = 0;
  }
  if(_ibo) {
    glDeleteBuffers(1, &_ibo);
    _ibo = 0;
  }
}

// Loads an OFF mesh file. See https://en.wikipedia.org/wiki/OFF_(file_format)
void loadOFF(const std::string &filename, std::shared_ptr<Mesh> meshPtr)
{
  std::cout << " > Start loading mesh <" << filename << ">" << std::endl;
  meshPtr->clear();
  std::ifstream in(filename.c_str());
  if(!in)
    throw std::ios_base::failure("[Mesh Loader][loadOFF] Cannot open " + filename);
  std::string offString;
  unsigned int sizeV, sizeT, tmp;
  in >> offString >> sizeV >> sizeT >> tmp;
  auto &P = meshPtr->vertexPositions();
  auto &T = meshPtr->triangleIndices();
  P.resize(sizeV);
  T.resize(sizeT);
  size_t tracker = (sizeV + sizeT)/20;
  std::cout << " > [" << std::flush;
  for(unsigned int i=0; i<sizeV; ++i) {
    if(i % tracker == 0)
      std::cout << "-" << std::flush;
    in >> P[i][0] >> P[i][1] >> P[i][2];
  }
  int s;
  for(unsigned int i=0; i<sizeT; ++i) {
    if((sizeV + i) % tracker == 0)
      std::cout << "-" << std::flush;
    in >> s;
    for(unsigned int j=0; j<3; ++j)
      in >> T[i][j];
  }
  std::cout << "]" << std::endl;
  in.close();
  meshPtr->vertexNormals().resize(P.size(), glm::vec3(0.f, 0.f, 1.f));
  meshPtr->vertexTexCoords().resize(P.size(), glm::vec2(0.f, 0.f));
  meshPtr->recomputePerVertexNormals();
  meshPtr->recomputePerVertexTextureCoordinates();
  std::cout << " > Mesh <" << filename << "> loaded" <<  std::endl;
}


// ------------------------------------------------ PROJET -------------------------------------------------------------

const float Mesh::radialCurvature(glm::vec3 camPos, int i) {
  std::pair<std::vector<float>, std::vector<glm::vec3>> principalCurvs = computePrincipalCurvatures(i);
  float k1 = principalCurvs.first[0];
  float k2 = principalCurvs.first[1];
  glm::vec3 direction1 = principalCurvs.second[0];
  glm::vec3 direction2 = principalCurvs.second[1];
  //Calcul de w
  glm::vec3 n = _vertexNormals[i];
  glm::vec3 v = camPos - _vertexPositions[i];
  //Calcul projection v sur plan tangent
  glm::vec3 w;

  w = glm::dot(v,direction1)*direction1 + glm::dot(v,direction2)*direction2;

  float psi = glm::acos(glm::dot(glm::normalize(w), glm::normalize(direction1)));
  //Ce calcul est basé sur TEST (et est probablement faux :D)

  float kr = k1* pow(cos(psi), 2.f) + k2 * pow(sin(psi), 2.f);
  return kr;
}

//Suggestive contours limitations
const bool Mesh::gradientLimitation(glm::vec3 position) {
//TODO
  return true;
}
  
const bool Mesh::tresholdLimitation(glm::vec3 position) {
//TODO
  return true;
}

//Finding suggestive contours
const bool Mesh::isSuggestiveContour(glm::vec3 camPos, int i) {
  if(isFront(camPos, i)) {
    float kr = radialCurvature(camPos, i);
    if(kr >=0.0001) {
      return false;
    }
  /*else {
    if(gradientLimitation(position) && tresholdLimitation(position)) {
      return true;
    }
  }
  return false;*/
    return true;
    }
  return false;
  }


// --------- Trouver courbure principal ---------

/**
 * @param i The index of the considered vertex
*/
const glm::mat3 Mesh::computeM(int i) {
  glm::mat3 M;

  //Calcul des wij
  std::vector<float> wijList = computeWijList(i);  

  std::vector<int> neighbours = computeNeighbours(i);
  for(int j : neighbours) {

    //Calcul de Tij
    glm::mat3 NNt;
    glm::vec3 N = _vertexNormals[i];
    NNt[0] = glm::vec3(N[0]*N[0] , N[0]*N[1], N[0]*N[2]);
    NNt[1] = glm::vec3(N[1]*N[0] , N[1]*N[1], N[1]*N[2]);
    NNt[2] = glm::vec3(N[2]*N[0] , N[2]*N[1], N[2]*N[2]);

    glm::vec3 vji = _vertexPositions[i] - _vertexPositions[j];

    glm::vec3 T = (glm::mat3(1.0f) - NNt)*vji;
    T = glm::normalize(T);

    //Calcul de Kij
    glm::vec3 normdiffv = glm::normalize(vji);
    float Kij = 2* glm::dot(N, normdiffv );

    //Calcul de M
    glm::mat3 TTt;
    TTt[0] = glm::vec3(T[0]*T[0] , T[0]*T[1], T[0]*T[2]);
    TTt[1] = glm::vec3(T[1]*T[0] , T[1]*T[1], T[1]*T[2]);
    TTt[2] = glm::vec3(T[2]*T[0] , T[2]*T[1], T[2]*T[2]);

    glm::mat3 Mij = wijList[j]*Kij*TTt;
    M+=Mij;
  }

  return M;
}

//Find triangles that are incident to both vi and vj
std::vector<int> Mesh::incTriangles(int i, int j) {
  std::vector<int> res;
  for(int k = 0; k< _triangleIndices.size() ; k++) {
    glm::uvec3 triangle = _triangleIndices[k];
    
    if(triangle[0]==i) {
      if (triangle[1]==j || triangle[2]==j) {
        res.push_back(k);
      }
    }
    else if(triangle[1]==i) {
      if(triangle[0]==j || triangle[2]==j) {
        res.push_back(k);
      }
    }
    
    else if(triangle[2]==i) {
      if(triangle[0]==j || triangle[1]==j) {
        res.push_back(k);
      }
    }
  }
  return res;
}

const std::vector<float> Mesh::computeWijList(int i) {
  std::vector<float> wijList;
  for(int j = 0; j< _vertexPositions.size() ; j++) {
    wijList.push_back(0.f);
  }

  float wijSum=0;

  std::vector<int> neighbours = computeNeighbours(i);
  for(int j : neighbours) {
    if(j!=i) {
      std::vector<int> incTri = incTriangles(i, j);
      float wij = 0.f;
      for(int k : incTri) {
        glm::uvec3 triangle = _triangleIndices[k];
        float lengthA = (_vertexPositions[triangle[2]] - _vertexPositions[triangle[1]]).length();
        float lengthB = (_vertexPositions[triangle[1]] - _vertexPositions[triangle[0]]).length();
        float lengthC = (_vertexPositions[triangle[0]] - _vertexPositions[triangle[2]]).length();

        float p = (lengthA+lengthB+lengthC)/2.f;

        float surface = std::sqrt(p*(p-lengthA)*(p-lengthB)*(p-lengthC));

        wij+= surface;
      }
      wijList[j] = wij;
      wijSum+=wij;
    }
  }

  for(int j = 0; j< _vertexPositions.size() ; j++) {
    wijList[j]/= wijSum;
  }

  return wijList;
}

const std::vector<int> Mesh::computeNeighbours(int i) {
  std::vector<int> neighbours;

  std::vector<int> trianglesWithI;
  for(int k = 0; k<_triangleIndices.size() ; k++ ){
    glm::uvec3 triangle = _triangleIndices[k];
    if(triangle[0]==i || triangle[1]==i || triangle[2] ==i) {
      trianglesWithI.push_back(k);
    }
  }
  
  for(int j = 0; j< _vertexPositions.size() ; j++ ) {
    for(int k : trianglesWithI) {
      glm::uvec3 triangle = _triangleIndices[k];
    
      if(triangle[0]==i) {
        if(triangle[1]==j || triangle[2]==j) {
          neighbours.push_back(j);
        }
      }
      
      else if(triangle[1]==i) {
        if(triangle[0]==j || triangle[2]==j) {
          neighbours.push_back(j);
        }
      }
     
      else if(triangle[2]==i) {
        if(triangle[0]==j || triangle[1]==j) {
          neighbours.push_back(j);
        }
      }
      
    }
  }
  return neighbours;
}

const std::pair<std::vector<float>, std::vector<glm::vec3> > Mesh::computePrincipalCurvatures(int i) {
  glm::mat3 M = computeM(i);

  glm::vec3 N = _vertexNormals[i];
  glm::vec3 E1 = glm::vec3(1.f,0.f,0.f);

  float sign =1;
  if( (N-E1).length()>(N+E1).length()) {
    sign = -1;
  }

  glm::vec3 W = glm::normalize(E1 + sign*N);

  glm::mat3 WWt;
  WWt[0] = glm::vec3(W[0]*W[0] , W[0]*W[1], W[0]*W[2]);
  WWt[1] = glm::vec3(W[1]*W[0] , W[1]*W[1], W[1]*W[2]);
  WWt[2] = glm::vec3(W[2]*W[0] , W[2]*W[1], W[2]*W[2]);

  glm::mat3 Q = glm::mat3(1.f) - 2.f*WWt;

  glm::mat3 diagM = glm::transpose(Q)*M*Q;

  glm::mat2 restrM;
  restrM[0] = glm::vec2(diagM[1][1], diagM[1][2]);
  restrM[1] = glm::vec2(diagM[2][1], diagM[2][2]);

  std::pair<std::vector<float>, std::vector<glm::vec3> > res = diagonalize(restrM, Q);
  
  
  std::cout << "Finish computing principal curvature at vertex " << i << std::endl;


  return res;
  
}

const std::pair<std::vector<float>, std::vector<glm::vec3> > Mesh::diagonalize(glm::mat2 mat, glm::mat3 Q) {
  glm::vec2 eigVec1;
  glm::vec2 eigVec2;
  float eigVal1;
  float eigVal2;

  if(mat[0][1]==0) {
    eigVec1 = glm::vec2(1.f, 0.f);
    eigVec2 = glm::vec2(0.f, 1.f);
    eigVal1 = mat[0][0];
    eigVal2 = mat[1][1];
  }

  else {
    float delta = pow((-mat[0][0] + mat [1][1]),2.f) + 4.f*pow(mat[0][1],2.f);
    float m = (-mat[0][0] + mat[1][1] + std::sqrt(delta))/(2.f*mat[1][0]);

    eigVec1 = glm::vec2(1.f, m);
    eigVec1/= std::sqrt(1+pow(m,2.f));
    eigVec2 = glm::vec2(-m, 1.f);
    eigVec2/= std::sqrt(1+pow(m,2.f));

    eigVal1 = (mat[0][0] + 2.f*mat[1][0]*m + mat[1][1]*pow(m,2.f))/(1+pow(m,2.f));
    eigVal2 = (mat[1][1] - 2.f*mat[1][0]*m + mat[0][0]*pow(m,2.f))/(1+pow(m,2.f));

  }

  // ------------ TEST ------------

  glm::vec3 eigVec13D = glm::inverse(glm::transpose(Q))*glm::vec3(0, eigVec1[0], eigVec1[1])*glm::inverse(Q);
  glm::vec3 eigVec23D = glm::inverse(glm::transpose(Q))*glm::vec3(0, eigVec1[0], eigVec1[1])*glm::inverse(Q);

  // ---------- FIN TEST ----------

  std::vector<float> eigVal{eigVal1, eigVal2};
  std::vector<glm::vec3> eigVec{eigVec13D, eigVec23D};



  std::pair<std::vector<float>, std::vector<glm::vec3> > res (
    eigVal, eigVec
  );

  return res;
}

//Compute all suggestive contours
const void Mesh::computeSuggContours(glm::vec3 camPos) {
  for(int i= 0; i< _vertexPositions.size(); i++) {
    if(isSuggestiveContour(camPos, i)) {
      _isSuggContour.push_back(1.f);
    }
    else {
      _isSuggContour.push_back(0.f);
    }
  }
}


const bool Mesh::isFront(glm::vec3 camPos, int i) {
  if(glm::dot(_vertexNormals[i], camPos- _vertexPositions[i]) <= 0) {
    return true;
  }
  return false;
}
