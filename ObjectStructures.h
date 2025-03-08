// ObjectStructures.h
#ifndef OBJECT_STRUCTURES_H
#define OBJECT_STRUCTURES_H

struct SceneVertex3D {
    float x, y, z;
};

struct ObjectBoundingBox {
    SceneVertex3D min; // Minimum corner
    SceneVertex3D max; // Maximum corner
};

#endif // OBJECT_STRUCTURES_H
