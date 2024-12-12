import yaml
import numpy as np
import pinocchio as pin
from pathlib import Path
from hppfcl import Sphere, Box, Cylinder, Capsule


class ObstacleParamsParser:
    def add_collisions(
        self, cmodel: pin.GeometryModel, yaml_file: Path
    ) -> pin.GeometryModel:
        new_cmodel = cmodel.copy()

        with open(str(yaml_file), "r") as file:
            params = yaml.safe_load(file)

        for obstacle_name in params:
            if obstacle_name == "collision_pairs":
                continue
            obstacle_config = params[obstacle_name]

            obstacle_type = obstacle_config.get("type")
            translation_vect = obstacle_config.get("translation", [])

            if not translation_vect:
                print(
                    f"No obstacle translation declared for the obstacle named: {obstacle_name}."
                )
                return cmodel.copy()

            translation = np.array(translation_vect).reshape(3)

            rotation_vect = obstacle_config.get("rotation", [])
            if not rotation_vect:
                print(
                    f"No obstacle rotation declared for the obstacle named: {obstacle_name}."
                )
                return cmodel.copy()

            rotation = np.array(rotation_vect).reshape(4)

            geometry = None
            if obstacle_type == "sphere":
                radius = obstacle_config.get("radius")
                if radius:
                    geometry = Sphere(radius)
                else:
                    print(
                        f"No dimension or wrong dimensions in the config for the obstacle named: {obstacle_name}."
                    )
                    return cmodel.copy()
            elif obstacle_type == "box":
                x = obstacle_config.get("x")
                y = obstacle_config.get("y")
                z = obstacle_config.get("z")
                if x and y and z:
                    geometry = Box(x, y, z)
                else:
                    print(
                        f"No dimension or wrong dimensions in the  config for the obstacle named: {obstacle_name}."
                    )
                    return cmodel.copy()
            elif obstacle_type == "cylinder":
                radius = obstacle_config.get("radius")
                half_length = obstacle_config.get("halfLength")
                if radius and half_length:
                    geometry = Cylinder(radius, half_length)
                else:
                    print(
                        f"No dimension or wrong dimensions in the  config for the obstacle named: {obstacle_name}."
                    )
                    return cmodel.copy()
            elif obstacle_type == "capsule":
                radius = obstacle_config.get("radius")
                half_length = obstacle_config.get("halfLength")
                if radius and half_length:
                    geometry = Capsule(radius, half_length)
                else:
                    print(
                        f"No dimension or wrong dimensions in the  config for the obstacle named: {obstacle_name}."
                    )
                    return cmodel.copy()
            else:
                print(
                    f"No type or wrong type in the config for the obstacle named: {obstacle_name}."
                )
                return cmodel.copy()
            obstacle_pose = pin.XYZQUATToSE3(np.concatenate([translation, rotation]))
            obstacle_pose.translation = translation
            obstacle = pin.GeometryObject(obstacle_name, 0, 0, geometry, obstacle_pose)
            new_cmodel.addGeometryObject(obstacle)

        collision_pairs = params.get("collision_pairs", [])
        if collision_pairs:
            for pair in collision_pairs:
                if len(pair) == 2:
                    name_object1, name_object2 = pair
                    if new_cmodel.existGeometryName(
                        name_object1
                    ) and new_cmodel.existGeometryName(name_object2):
                        new_cmodel = self.add_collision_pair(
                            new_cmodel, name_object1, name_object2
                        )
                    else:
                        print(
                            f"Object {name_object1} or {name_object2} does not exist in the collision model."
                        )
                else:
                    print(f"Invalid collision pair: {pair}.")
        else:
            print("No collision pairs defined.")

        return new_cmodel

    def add_collision_pair(
        self, cmodel: pin.GeometryModel, name_object1: str, name_object2: str
    ) -> pin.GeometryModel:
        if cmodel.existGeometryName(name_object1) and cmodel.existGeometryName(
            name_object2
        ):
            object1_id = cmodel.getGeometryId(name_object1)
            object2_id = cmodel.getGeometryId(name_object2)
            cmodel.addCollisionPair(pin.CollisionPair(object1_id, object2_id))
        else:
            print(
                f"Object ID not found for collision pair: {object1_id} and {object2_id}."
            )
        return cmodel

    def transform_model_into_capsules(
        self, model: pin.GeometryModel
    ) -> pin.GeometryModel:
        """Modifying the collision model to transform the spheres/cylinders into capsules which makes it easier to have a fully constrained robot."""
        model_copy = model.copy()

        # Going through all the geometry objects in the collision model
        cylinders_name = [
            obj.name
            for obj in model_copy.geometryObjects
            if isinstance(obj.geometry, Cylinder)
        ]
        for cylinder_name in cylinders_name:
            basename = cylinder_name.rsplit("_", 1)[0]
            col_index = int(cylinder_name.rsplit("_", 1)[1])
            sphere1_name = basename + "_" + str(col_index + 1)
            sphere2_name = basename + "_" + str(col_index + 2)
            if not model_copy.existGeometryName(
                sphere1_name
            ) or not model_copy.existGeometryName(sphere2_name):
                continue

            # Sometimes for one joint there are two cylinders, which need to be defined by two capsules for the same link.
            # Hence the name convention here.
            capsules_already_existing = [
                obj.name
                for obj in model_copy.geometryObjects
                if (basename in obj.name and "capsule" in obj.name)
            ]
            capsule_name = basename + "_capsule_" + str(len(capsules_already_existing))
            geom_object = model_copy.geometryObjects[
                model_copy.getGeometryId(cylinder_name)
            ]
            placement = geom_object.placement
            parentJoint = geom_object.parentJoint
            parentFrame = geom_object.parentFrame
            geometry = geom_object.geometry
            geom = pin.GeometryObject(
                capsule_name,
                parentFrame,
                parentJoint,
                Capsule(geometry.radius, geometry.halfLength),
                placement,
            )
            RED = np.array([249, 136, 126, 125]) / 255
            geom.meshColor = RED
            model_copy.removeGeometryObject(cylinder_name)
            model_copy.removeGeometryObject(sphere1_name)
            model_copy.removeGeometryObject(sphere2_name)
            model_copy.addGeometryObject(geom)

        # Purge all non capsule and non sphere geometry
        none_convex_object_names = [
            obj.name
            for obj in model_copy.geometryObjects
            if not (
                isinstance(obj.geometry, Capsule) or isinstance(obj.geometry, Sphere)
            )
        ]
        for none_convex_object_name in none_convex_object_names:
            model_copy.removeGeometryObject(none_convex_object_name)

        # Return the copy of the model.
        return model_copy

    def add_self_collision(
        self, rmodel: pin.Model, rcmodel: pin.GeometryModel, srdf: Path = Path()
    ) -> pin.GeometryModel:
        rcmodel.addAllCollisionPairs()
        if srdf.is_file():
            pin.removeCollisionPairs(rmodel, rcmodel, str(srdf))
        return rcmodel
