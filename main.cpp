#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <memory>

// Definir M_PI si no está definido
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Vector 3D básico
struct Vec3 {
    float x, y, z;
    
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float t) const { return Vec3(x * t, y * t, z * t); }
    Vec3 operator/(float t) const { return Vec3(x / t, y / t, z / t); }
    
    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const { 
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); 
    }
    
    float length() const { return sqrt(x * x + y * y + z * z); }
    Vec3 normalize() const { 
        float len = length(); 
        return len > 0 ? *this / len : Vec3(0, 0, 0); 
    }
    
    Vec3 reflect(const Vec3& normal) const {
        return *this - normal * 2.0f * this->dot(normal);
    }

    // Add this new operator after the other operators
    friend Vec3 operator*(float t, const Vec3& v) {
        return v * t;  // Reuse the existing v * t operator
    }
};

// Color RGB
struct Color {
    float r, g, b;
    
    Color(float r = 0, float g = 0, float b = 0) : r(r), g(g), b(b) {}
    
    Color operator+(const Color& c) const { return Color(r + c.r, g + c.g, b + c.b); }
    Color operator*(float t) const { return Color(r * t, g * t, b * t); }
    Color operator*(const Color& c) const { return Color(r * c.r, g * c.g, b * c.b); }
    
    Color clamp() const {
        return Color(std::min(1.0f, std::max(0.0f, r)),
                    std::min(1.0f, std::max(0.0f, g)),
                    std::min(1.0f, std::max(0.0f, b)));
    }
    
    // Convertir a entero para guardar imagen
    int toInt() const {
        int ir = (int)(255 * clamp().r);
        int ig = (int)(255 * clamp().g);
        int ib = (int)(255 * clamp().b);
        return (ir << 16) | (ig << 8) | ib;
    }
};

// Material con todas las propiedades requeridas
struct Material {
    Color albedo;           // Color difuso
    Color specular;         // Color especular
    float reflectivity;     // Qué tanto refleja (0-1)
    float transparency;     // Qué tanto es transparente (0-1)
    float refraction_index; // Índice de refracción
    float roughness;        // Rugosidad de la superficie
    int texture_id;         // ID de textura (-1 si no tiene)
    
    Material(Color albedo = Color(0.8, 0.8, 0.8), 
             Color specular = Color(0.2, 0.2, 0.2),
             float reflectivity = 0.0f, 
             float transparency = 0.0f,
             float refraction_index = 1.0f,
             float roughness = 1.0f,
             int texture_id = -1)
        : albedo(albedo), specular(specular), reflectivity(reflectivity), 
          transparency(transparency), refraction_index(refraction_index),
          roughness(roughness), texture_id(texture_id) {}
};

// Estructura para intersecciones
struct Intersection {
    bool hit;
    float distance;
    Vec3 point;
    Vec3 normal;
    Material material;
    Vec3 tex_coords; // Coordenadas de textura (u, v, 0)
    
    Intersection() : hit(false), distance(0) {}
    Intersection(float dist, Vec3 p, Vec3 n, Material m, Vec3 tex = Vec3())
        : hit(true), distance(dist), point(p), normal(n), material(m), tex_coords(tex) {}
};

// Rayo
struct Ray {
    Vec3 origin;
    Vec3 direction;
    
    Ray(Vec3 o, Vec3 d) : origin(o), direction(d.normalize()) {}
    
    Vec3 at(float t) const { return origin + direction * t; }
};

// Clase base para objetos
class Object {
public:
    Material material;
    
    Object(Material m) : material(m) {}
    virtual ~Object() {}
    virtual Intersection intersect(const Ray& ray) const = 0;
};

// Cubo (la estrella del show para este proyecto)
class Cube : public Object {
public:
    Vec3 center;
    float size;
    
    Cube(Vec3 c, float s, Material m) : Object(m), center(c), size(s) {}
    
    Intersection intersect(const Ray& ray) const override {
        Vec3 min_bound = center - Vec3(size/2, size/2, size/2);
        Vec3 max_bound = center + Vec3(size/2, size/2, size/2);
        
        float t_min = (min_bound.x - ray.origin.x) / ray.direction.x;
        float t_max = (max_bound.x - ray.origin.x) / ray.direction.x;
        
        if (t_min > t_max) std::swap(t_min, t_max);
        
        float ty_min = (min_bound.y - ray.origin.y) / ray.direction.y;
        float ty_max = (max_bound.y - ray.origin.y) / ray.direction.y;
        
        if (ty_min > ty_max) std::swap(ty_min, ty_max);
        
        if (t_min > ty_max || ty_min > t_max) return Intersection();
        
        t_min = std::max(t_min, ty_min);
        t_max = std::min(t_max, ty_max);
        
        float tz_min = (min_bound.z - ray.origin.z) / ray.direction.z;
        float tz_max = (max_bound.z - ray.origin.z) / ray.direction.z;
        
        if (tz_min > tz_max) std::swap(tz_min, tz_max);
        
        if (t_min > tz_max || tz_min > t_max) return Intersection();
        
        t_min = std::max(t_min, tz_min);
        
        if (t_min < 0) return Intersection();
        
        Vec3 hit_point = ray.at(t_min);
        Vec3 normal = calculateNormal(hit_point, min_bound, max_bound);
        Vec3 tex_coords = calculateTexCoords(hit_point, normal);
        
        return Intersection(t_min, hit_point, normal, material, tex_coords);
    }
    
private:
    Vec3 calculateNormal(const Vec3& hit_point, const Vec3& min_bound, const Vec3& max_bound) const {
        Vec3 center_to_hit = hit_point - center;
        Vec3 abs_center = Vec3(abs(center_to_hit.x), abs(center_to_hit.y), abs(center_to_hit.z));
        
        float max_component = std::max({abs_center.x, abs_center.y, abs_center.z});
        
        if (abs_center.x == max_component) {
            return Vec3(center_to_hit.x > 0 ? 1 : -1, 0, 0);
        } else if (abs_center.y == max_component) {
            return Vec3(0, center_to_hit.y > 0 ? 1 : -1, 0);
        } else {
            return Vec3(0, 0, center_to_hit.z > 0 ? 1 : -1);
        }
    }
    
    Vec3 calculateTexCoords(const Vec3& hit_point, const Vec3& normal) const {
        // Mapear coordenadas de textura basadas en la cara del cubo
        Vec3 local = hit_point - center;
        
        if (abs(normal.x) > 0.5) { // Cara X
            return Vec3((local.z + size/2) / size, (local.y + size/2) / size, 0);
        } else if (abs(normal.y) > 0.5) { // Cara Y
            return Vec3((local.x + size/2) / size, (local.z + size/2) / size, 0);
        } else { // Cara Z
            return Vec3((local.x + size/2) / size, (local.y + size/2) / size, 0);
        }
    }
};

// Esfera (para variedad)
class Sphere : public Object {
public:
    Vec3 center;
    float radius;
    
    Sphere(Vec3 c, float r, Material m) : Object(m), center(c), radius(r) {}
    
    Intersection intersect(const Ray& ray) const override {
        Vec3 oc = ray.origin - center;
        float a = ray.direction.dot(ray.direction);
        float b = 2.0f * oc.dot(ray.direction);
        float c = oc.dot(oc) - radius * radius;
        
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) return Intersection();
        
        float t = (-b - sqrt(discriminant)) / (2.0f * a);
        if (t < 0.001f) { // Evitar self-intersection
            t = (-b + sqrt(discriminant)) / (2.0f * a);
            if (t < 0.001f) return Intersection();
        }
        
        Vec3 hit_point = ray.at(t);
        Vec3 normal = (hit_point - center).normalize();
        
        return Intersection(t, hit_point, normal, material);
    }
};

// Cámara
struct Camera {
    Vec3 position;
    Vec3 target;
    Vec3 up;
    float fov;
    float aspect_ratio;
    
    Camera(Vec3 pos, Vec3 tgt, Vec3 u, float f, float ar) 
        : position(pos), target(tgt), up(u), fov(f), aspect_ratio(ar) {}
    
    Ray getRay(float u, float v) const {
        Vec3 w = (position - target).normalize();
        Vec3 u_vec = up.cross(w).normalize();
        Vec3 v_vec = w.cross(u_vec);
        
        float theta = fov * M_PI / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect_ratio * half_height;
        
        Vec3 lower_left = position - half_width * u_vec - half_height * v_vec - w;
        Vec3 horizontal = u_vec * (2.0f * half_width);
        Vec3 vertical = v_vec * (2.0f * half_height);
        
        Vec3 direction = lower_left + u * horizontal + v * vertical - position;
        return Ray(position, direction);
    }
};

// Raytracer principal
class Raytracer {
private:
    std::vector<std::unique_ptr<Object>> objects;
    Camera camera;
    int width, height;
    int max_depth;
    
public:
    Raytracer(int w, int h, Camera cam) 
        : width(w), height(h), camera(cam), max_depth(5) {}
    
    void addObject(std::unique_ptr<Object> obj) {
        objects.push_back(std::move(obj));
    }
    
    Color castRay(const Ray& ray, int depth) const {
        if (depth <= 0) return Color(0, 0, 0);
        
        Intersection closest;
        closest.distance = std::numeric_limits<float>::max();
        
        // Encontrar la intersección más cercana
        for (const auto& obj : objects) {
            Intersection intersection = obj->intersect(ray);
            if (intersection.hit && intersection.distance < closest.distance) {
                closest = intersection;
            }
        }
        
        if (!closest.hit) {
            return getSkyboxColor(ray.direction);
        }
        
        return shade(ray, closest, depth);
    }
    
    Color shade(const Ray& ray, const Intersection& intersection, int depth) const {
        Color result = intersection.material.albedo * 0.1f; // Luz ambiente
        
        // Reflexión
        if (intersection.material.reflectivity > 0) {
            Vec3 reflect_dir = ray.direction.reflect(intersection.normal);
            Ray reflect_ray(intersection.point + intersection.normal * 0.001f, reflect_dir);
            Color reflect_color = castRay(reflect_ray, depth - 1);
            result = result + reflect_color * intersection.material.reflectivity;
        }
        
        return result.clamp();
    }
    
    Color getSkyboxColor(const Vec3& direction) const {
        // Gradiente simple de cielo
        float t = 0.5f * (direction.y + 1.0f);
        return Color(0.5f, 0.7f, 1.0f) * (1.0f - t) + Color(1.0f, 1.0f, 1.0f) * t;
    }
    
    void render() {
        std::vector<Color> framebuffer(width * height);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float u = float(x) / float(width - 1);
                float v = float(height - 1 - y) / float(height - 1);
                
                Ray ray = camera.getRay(u, v);
                Color color = castRay(ray, max_depth);
                
                framebuffer[y * width + x] = color;
            }
            
            // Mostrar progreso
            if (y % 50 == 0) {
                std::cout << "Renderizando línea " << y << "/" << height << std::endl;
            }
        }
        
        saveImage(framebuffer, "output.ppm");
    }
    
private:
    void saveImage(const std::vector<Color>& framebuffer, const std::string& filename) {
        std::ofstream file(filename);
        file << "P3\n" << width << " " << height << "\n255\n";
        
        for (const Color& color : framebuffer) {
            Color c = color.clamp();
            int r = int(255 * c.r);
            int g = int(255 * c.g);
            int b = int(255 * c.b);
            file << r << " " << g << " " << b << "\n";
        }
    }
};

int main() {
    const int WIDTH = 800;
    const int HEIGHT = 600;
    
    // Crear cámara
    Camera camera(Vec3(5, 3, 8), Vec3(0, 0, 0), Vec3(0, 1, 0), 60.0f, float(WIDTH) / HEIGHT);
    
    // Crear raytracer
    Raytracer raytracer(WIDTH, HEIGHT, camera);
    
    // Crear escena con diferentes materiales
    
    // Material piedra
    Material stone(Color(0.5f, 0.5f, 0.5f), Color(0.1f, 0.1f, 0.1f), 0.1f);
    
    // Material metal
    Material metal(Color(0.7f, 0.7f, 0.8f), Color(0.8f, 0.8f, 0.9f), 0.8f);
    
    // Material vidrio
    Material glass(Color(0.9f, 0.9f, 1.0f), Color(0.9f, 0.9f, 1.0f), 0.1f, 0.9f, 1.5f);
    
    // Agregar objetos a la escena
    raytracer.addObject(std::make_unique<Cube>(Vec3(0, 0, 0), 1.0f, stone));
    raytracer.addObject(std::make_unique<Cube>(Vec3(2, 0, 0), 1.0f, metal));
    raytracer.addObject(std::make_unique<Sphere>(Vec3(-2, 0, 0), 0.5f, glass));
    
    // Base
    raytracer.addObject(std::make_unique<Cube>(Vec3(0, -1.5f, 0), 6.0f, stone));
    
    std::cout << "Iniciando renderizado..." << std::endl;
    raytracer.render();
    std::cout << "¡Renderizado completo! Archivo guardado como output.ppm" << std::endl;
    
    return 0;
}
