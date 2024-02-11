#ifndef MESH_H
#define MESH_H

#include <glad/glad.h>
#include <vector>
#include <memory>
#include <string>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

class Mesh {
public:
  virtual ~Mesh();

  const std::vector<glm::vec3> &vertexPositions() const { return _vertexPositions; }
  std::vector<glm::vec3> &vertexPositions() { return _vertexPositions; }

  const std::vector<glm::vec3> &vertexNormals() const { return _vertexNormals; }
  std::vector<glm::vec3> &vertexNormals() { return _vertexNormals; }

  const std::vector<glm::vec2> &vertexTexCoords() const { return _vertexTexCoords; }
  std::vector<glm::vec2> &vertexTexCoords() { return _vertexTexCoords; }

  const std::vector<glm::uvec3> &triangleIndices() const { return _triangleIndices; }
  std::vector<glm::uvec3> &triangleIndices() { return _triangleIndices; }


  // ------------- finding contours and suggestive contours -------------
  
  //Curvatures
  const glm::mat3 computeM(int i);
  const std::vector<float> computeWijList(int i);
  const std::vector<int> computeNeighbours(int i);
  std::vector<int> incTriangles(int i, int j);
  const std::pair<std::vector<float>, std::vector<glm::vec3> > diagonalize(glm::mat2 mat, glm::mat3 Q);

  const std::pair<std::vector<float>, std::vector<glm::vec3> > computePrincipalCurvatures(int i);

  const float radialCurvature(glm::vec3 camPos, int i);

  //Suggestive contours limitations
  const bool gradientLimitation(glm::vec3 position);
  const bool tresholdLimitation(glm::vec3 position);

  //Finding suggestive contours
  const bool isSuggestiveContour(glm::vec3 camPos, int i);
  const void computeSuggContours(glm::vec3 camPos);
  //Finding contours
  //const bool isContour(std::shared_ptr<Camera>, glm::vec3 position);

  //Optimisation : compute the curvatures only for the vertex that the camera can see
  const bool isFront(glm::vec3 camPos, int i);

  // --------------------------------------------------------------------

  /// Compute the parameters of a sphere which bounds the mesh
  void computeBoundingSphere(glm::vec3 &center, float &radius) const;

  void recomputePerVertexNormals(bool angleBased = false);
  void recomputePerVertexTextureCoordinates( );

  void init(glm::vec3 camPos);
  void initOldGL();
  void render();
  void clear();

  void addPlan(float square_half_side = 1.0f);

private:
  std::vector<glm::vec3> _vertexPositions;
  std::vector<glm::vec3> _vertexNormals;
  std::vector<glm::vec2> _vertexTexCoords;
  std::vector<glm::uvec3> _triangleIndices;

  GLuint _vao = 0;
  GLuint _posVbo = 0;
  GLuint _normalVbo = 0;
  GLuint _texCoordVbo = 0;
  GLuint _ibo = 0;

  // --------- finding contours and suggestive contours ---------
  GLuint _isSuggContourVbo = 0;
  std::vector<float> _isSuggContour;
  // ------------------------------------------------------------

};

// utility: loader
void loadOFF(const std::string &filename, std::shared_ptr<Mesh> meshPtr);

#endif  // MESH_H
