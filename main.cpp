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

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"


// Vector 3D
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

    friend Vec3 operator*(float t, const Vec3& v) {
        return v * t;
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
    
    int toInt() const {
        int ir = (int)(255 * clamp().r);
        int ig = (int)(255 * clamp().g);
        int ib = (int)(255 * clamp().b);
        return (ir << 16) | (ig << 8) | ib;
    }
};

// Clase base para texturas
class Texture {
public:
    virtual ~Texture() {}
    virtual Color sample(float u, float v) const = 0;
};

// Textura solida (un solo color)
class SolidTexture : public Texture {
private:
    Color color;
    
public:
    SolidTexture(Color c) : color(c) {}
    
    Color sample(float u, float v) const override {
        return color;
    }
};

// Textura de tablero de ajedrez
class CheckerTexture : public Texture {
private:
    Color color1, color2;
    float scale;
    
public:
    CheckerTexture(Color c1, Color c2, float s = 8.0f) 
        : color1(c1), color2(c2), scale(s) {}
    
    Color sample(float u, float v) const override {
        int checker_u = int(floor(u * scale));
        int checker_v = int(floor(v * scale));
        
        if ((checker_u + checker_v) % 2 == 0) {
            return color1;
        } else {
            return color2;
        }
    }
};

// Textura de ladrillos
class BrickTexture : public Texture {
private:
    Color brick_color, mortar_color;
    float brick_width, brick_height;
    float mortar_width;
    
public:
    BrickTexture(Color brick = Color(0.7f, 0.3f, 0.2f), 
                 Color mortar = Color(0.9f, 0.9f, 0.8f),
                 float bw = 0.3f, float bh = 0.15f, float mw = 0.02f)
        : brick_color(brick), mortar_color(mortar), 
          brick_width(bw), brick_height(bh), mortar_width(mw) {}
    
    Color sample(float u, float v) const override {
        u = u - floor(u);
        v = v - floor(v);
        
        int row = int(v / (brick_height + mortar_width));
        float row_v = v - row * (brick_height + mortar_width);
        
        float offset = (row % 2) * (brick_width + mortar_width) * 0.5f;
        float adjusted_u = u + offset;
        adjusted_u = adjusted_u - floor(adjusted_u);
        
        if (row_v < mortar_width || row_v > brick_height) {
            return mortar_color;
        }
        
        int col = int(adjusted_u / (brick_width + mortar_width));
        float col_u = adjusted_u - col * (brick_width + mortar_width);
        
        if (col_u < mortar_width) {
            return mortar_color;
        }
        
        return brick_color;
    }
};

// Textura de madera
class WoodTexture : public Texture {
private:
    Color light_wood, dark_wood;
    float grain_frequency;
    
public:
    WoodTexture(Color light = Color(0.8f, 0.6f, 0.3f),
                Color dark = Color(0.5f, 0.3f, 0.1f),
                float freq = 10.0f)
        : light_wood(light), dark_wood(dark), grain_frequency(freq) {}
    
    Color sample(float u, float v) const override {
        float center_u = 0.5f;
        float center_v = 0.5f;
        float dist = sqrt((u - center_u) * (u - center_u) + (v - center_v) * (v - center_v));
        
        float ring = sin(dist * grain_frequency + v * 20.0f) * 0.5f + 0.5f;
        
        return light_wood * ring + dark_wood * (1.0f - ring);
    }
};

//Textura de piedra
class StoneTexture : public Texture {
private:
    Color stone_light, stone_dark;
    
public:
    StoneTexture(Color light = Color(0.7f, 0.7f, 0.7f),
                 Color dark = Color(0.4f, 0.4f, 0.4f))
        : stone_light(light), stone_dark(dark) {}
    
    Color sample(float u, float v) const override {
        float noise1 = sin(u * 50.0f) * sin(v * 50.0f);
        float noise2 = sin(u * 100.0f + v * 80.0f) * 0.5f;
        float pattern = (noise1 + noise2) * 0.5f + 0.5f;
        
        return stone_light * pattern + stone_dark * (1.0f - pattern);
    }
};

// Textura de metal rayado
class MetalTexture : public Texture {
private:
    Color base_color, scratch_color;
    float scratch_frequency;
    
public:
    MetalTexture(Color base = Color(0.8f, 0.8f, 0.9f),
                 Color scratch = Color(0.6f, 0.6f, 0.7f),
                 float freq = 50.0f)
        : base_color(base), scratch_color(scratch), scratch_frequency(freq) {}
    
    Color sample(float u, float v) const override {
        float horizontal_scratch = sin(v * scratch_frequency) * 0.5f + 0.5f;
        float vertical_scratch = sin(u * scratch_frequency * 0.3f) * 0.3f + 0.7f;
        
        float scratch_intensity = horizontal_scratch * vertical_scratch;
        
        float noise = sin(u * 100.0f) * sin(v * 100.0f) * 0.1f + 0.9f;
        
        return base_color * scratch_intensity * noise + scratch_color * (1.0f - scratch_intensity);
    }
};

// Textura de hierba
class GrassTexture : public Texture {
private:
    Color grass_color1, grass_color2, dirt_color;
    
public:
    GrassTexture(Color grass1 = Color(0.2f, 0.6f, 0.2f),
                 Color grass2 = Color(0.3f, 0.8f, 0.3f),
                 Color dirt = Color(0.4f, 0.2f, 0.1f))
        : grass_color1(grass1), grass_color2(grass2), dirt_color(dirt) {}
    
    Color sample(float u, float v) const override {
        float pattern1 = sin(u * 20.0f) * sin(v * 20.0f);
        float pattern2 = sin(u * 50.0f + v * 30.0f) * 0.5f;
        float pattern3 = sin(u * 100.0f) * sin(v * 80.0f) * 0.3f;
        
        float grass_intensity = (pattern1 + pattern2 + pattern3) * 0.5f + 0.5f;
        
        Color grass_mix = grass_color1 * grass_intensity + grass_color2 * (1.0f - grass_intensity);
        
        if (grass_intensity < 0.2f) {
            return dirt_color * 0.7f + grass_mix * 0.3f;
        }
        
        return grass_mix;
    }
};

//Textura de hojas
class LeavesTexture : public Texture {
private:
    Color leaf_bright, leaf_dark;
    
public:
    LeavesTexture(Color bright = Color(0.3f, 0.8f, 0.2f),
                  Color dark = Color(0.1f, 0.5f, 0.1f))
        : leaf_bright(bright), leaf_dark(dark) {}
    
    Color sample(float u, float v) const override {
        float pixelated_u = floor(u * 16.0f) / 16.0f;
        float pixelated_v = floor(v * 16.0f) / 16.0f;
        
        float pattern = sin(pixelated_u * 100.0f) * sin(pixelated_v * 80.0f) * 0.5f + 0.5f;
        
        return leaf_bright * pattern + leaf_dark * (1.0f - pattern);
    }
};

// Textura de vidrio
class GlassTexture : public Texture {
private:
    Color base_tint;
    float pattern_scale;
    
public:
    GlassTexture(Color tint = Color(0.9f, 0.95f, 1.0f), float scale = 10.0f)
        : base_tint(tint), pattern_scale(scale) {}
    
    Color sample(float u, float v) const override {
        float pattern = sin(u * pattern_scale) * sin(v * pattern_scale) * 0.1f + 0.9f;
        return base_tint * pattern;
    }
};

class ImageTexture : public Texture {
private:
    std::vector<Color> image_data;
    int image_width, image_height;
    bool loaded;
    
public:
    ImageTexture(const std::string& filename) : loaded(false) {
        loadImage(filename);
    }
    
    bool loadImage(const std::string& filename) {
        int channels;
        unsigned char* data = stbi_load(filename.c_str(), &image_width, &image_height, &channels, 3);
        
        if (!data) {
            std::cout << "Error: No se pudo cargar la imagen " << filename << std::endl;
            return false;
        }
        
        std::cout << "Imagen cargada: " << filename << " (" << image_width << "x" << image_height << ")" << std::endl;
        
        // Convertir datos de imagen a nuestro formato Color
        image_data.resize(image_width * image_height);
        
        for (int i = 0; i < image_width * image_height; i++) {
            float r = data[i * 3 + 0] / 255.0f;
            float g = data[i * 3 + 1] / 255.0f;  
            float b = data[i * 3 + 2] / 255.0f;
            image_data[i] = Color(r, g, b);
        }
        
        stbi_image_free(data);
        loaded = true;
        return true;
    }
    
    Color sample(float u, float v) const override {
        if (!loaded || image_data.empty()) {
            return Color(1.0f, 0.0f, 1.0f); // Magenta si no se cargó
        }
        
        // Asegurar que u y v estén en rango [0,1]
        u = u - floor(u);
        v = v - floor(v);
        
        // Para texturas de Minecraft, usar nearest neighbor (sin interpolación)
        int x = int(u * image_width) % image_width;
        int y = int(v * image_height) % image_height;
        
        // Invertir Y porque las imágenes suelen estar al revés
        y = image_height - 1 - y;
        
        int index = y * image_width + x;
        return image_data[index];
    }
    
    // Versión con interpolación bilinear (opcional, para texturas más suaves)
    Color sampleBilinear(float u, float v) const {
        if (!loaded || image_data.empty()) {
            return Color(1.0f, 0.0f, 1.0f);
        }
        
        u = u - floor(u);
        v = v - floor(v);
        
        float fx = u * image_width;
        float fy = v * image_height;
        
        int x1 = int(fx) % image_width;
        int y1 = int(fy) % image_height;
        int x2 = (x1 + 1) % image_width;
        int y2 = (y1 + 1) % image_height;
        
        float dx = fx - floor(fx);
        float dy = fy - floor(fy);
        
        // Invertir Y
        y1 = image_height - 1 - y1;
        y2 = image_height - 1 - y2;
        
        Color c11 = image_data[y1 * image_width + x1];
        Color c21 = image_data[y1 * image_width + x2]; 
        Color c12 = image_data[y2 * image_width + x1];
        Color c22 = image_data[y2 * image_width + x2];
        
        // Interpolación bilinear
        Color c1 = c11 * (1.0f - dx) + c21 * dx;
        Color c2 = c12 * (1.0f - dx) + c22 * dx;
        
        return c1 * (1.0f - dy) + c2 * dy;
    }
    
    bool isLoaded() const { return loaded; }
    int getWidth() const { return image_width; }
    int getHeight() const { return image_height; }
};

// Manager de texturas
class TextureManager {
private:
    std::vector<std::unique_ptr<Texture>> textures;
    
public:
    TextureManager() {
        // CARGAR TEXTURAS DE IMAGEN
        loadImageTexture("textures/leaves.png");    // 0: Hojas de Minecraft
        loadImageTexture("textures/stone.png");     // 1: Piedra 
        loadImageTexture("textures/wood.png");      // 2: Tronco de madera
        loadImageTexture("textures/planks.png");    // 3: Tablones de madera
        loadImageTexture("textures/glass.jpg");     // 4: Vidrio
        loadImageTexture("textures/brick.png");     // 5: Ladrillos 
        loadImageTexture("textures/grass.png");     // 6: Hierba
    }
    
    int addTexture(std::unique_ptr<Texture> texture) {
        int id = textures.size();
        textures.push_back(std::move(texture));
        return id;
    }
    
    int loadImageTexture(const std::string& filename) {
        auto imageTexture = std::make_unique<ImageTexture>(filename);
        
        if (!imageTexture->isLoaded()) {
            std::cout << "Advertencia: No se pudo cargar " << filename << ", usando textura por defecto" << std::endl;
            // Usar textura por defecto si falla la carga
            return addTexture(std::make_unique<SolidTexture>(Color(0.5f, 0.5f, 0.5f)));
        }
        
        return addTexture(std::move(imageTexture));
    }
    
    Color sampleTexture(int texture_id, float u, float v) const {
        if (texture_id >= 0 && texture_id < textures.size()) {
            return textures[texture_id]->sample(u, v);
        }
        return Color(1.0f, 0.0f, 1.0f); // Magenta para texturas faltantes
    }
    
    size_t getTextureCount() const { return textures.size(); }
};

// Material con todas las propiedades requeridas
struct Material {
    Color albedo;
    Color specular;
    float reflectivity;
    float transparency;
    float refraction_index;
    float roughness;
    int texture_id;
    
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
    Vec3 tex_coords;
    
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

// Cubo
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
        Vec3 local = hit_point - center;
        
        if (abs(normal.x) > 0.5) {
            return Vec3((local.z + size/2) / size, (local.y + size/2) / size, 0);
        } else if (abs(normal.y) > 0.5) {
            return Vec3((local.x + size/2) / size, (local.z + size/2) / size, 0);
        } else {
            return Vec3((local.x + size/2) / size, (local.y + size/2) / size, 0);
        }
    }
};

// Esfera
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
        if (t < 0.001f) {
            t = (-b + sqrt(discriminant)) / (2.0f * a);
            if (t < 0.001f) return Intersection();
        }
        
        Vec3 hit_point = ray.at(t);
        Vec3 normal = (hit_point - center).normalize();
        
        return Intersection(t, hit_point, normal, material);
    }
};

// Camara
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
    TextureManager texture_manager; // Sistema de texturas
    
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
        Color base_color = intersection.material.albedo;
        
        // Aplicar textura si existe
        if (intersection.material.texture_id >= 0) {
            Color texture_color = texture_manager.sampleTexture(
                intersection.material.texture_id, 
                intersection.tex_coords.x, 
                intersection.tex_coords.y
            );
            base_color = base_color * texture_color;
        }
        
        Color result = base_color * 0.3f; // Luz ambiente
        
        // Luz direccional simple
        Vec3 light_dir = Vec3(0.5f, 1.0f, 0.3f).normalize();
        float light_intensity = std::max(0.0f, intersection.normal.dot(light_dir));
        result = result + base_color * light_intensity * 0.7f;
        
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
    
    // Crear camara para ver la escena
    Camera camera(Vec3(10, 10, 24), Vec3(-2, -2, 0), Vec3(0, 7, 0), 100.0f, float(WIDTH) / HEIGHT);

    // Crear raytracer
    Raytracer raytracer(WIDTH, HEIGHT, camera);
    
    Material leaves_material(
        Color(1.0f, 1.0f, 1.0f),
        Color(0.1f, 0.3f, 0.1f),
        0.02f,
        0.0f,
        1.0f,
        0.9f,
        0   // Updated ID for leaves texture
    );

    Material stone_material(
        Color(1.0f, 1.0f, 1.0f),
        Color(0.2f, 0.2f, 0.2f),
        0.05f,
        0.0f,
        1.0f,
        0.9f,
        1   // Updated ID for stone texture
    );

    Material wood_material(
        Color(1.0f, 1.0f, 1.0f),
        Color(0.2f, 0.1f, 0.05f),
        0.1f,
        0.0f,
        1.0f,
        0.8f,
        2   // Updated ID for wood texture
    );

    Material planks_material(
        Color(1.0f, 1.0f, 1.0f),
        Color(0.2f, 0.1f, 0.05f),
        0.1f,
        0.0f,
        1.0f,
        0.8f,
        3   // Updated ID for planks texture
    );

    Material glass_material(
        Color(1.0f, 1.0f, 1.0f),
        Color(0.95f, 0.98f, 1.0f),
        0.1f,
        0.85f,
        1.5f,
        0.02f,
        4   // Updated ID for glass texture
    );

    Material brick_material(
        Color(1.0f, 1.0f, 1.0f),
        Color(0.1f, 0.1f, 0.1f),
        0.05f,
        0.0f,
        1.0f,
        0.9f,
        5   // Updated ID for brick texture
    );
    
    Material grass_material(
        Color(1.0f, 1.0f, 1.0f),  // Using base color instead of texture
        Color(0.1f, 0.1f, 0.1f),
        0.02f,
        0.0f,
        1.0f,
        1.0f,
        6  // No texture for grass
    );

    // ================ CREAR DIORAMA ================
    
    // SUELO BASE - Hierba
    for (int x = -5; x <= 6; x++) {
        for (int z = -5; z <= 6; z++) {
            raytracer.addObject(std::make_unique<Cube>(Vec3(x * 2, 0, z * 2), 2.0f, grass_material));
        }
    }
    
    //Pilar Izquierdo frente (tronco)
    for (int y = 1; y <= 4; y++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(-8, y * 2, 2), 2.0f, wood_material));
    }

    //Pilar Derecho frente (tronco)
    for (int y = 1; y <= 4; y++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(2, y * 2, 2), 2.0f, wood_material));
    }

    //Pilar Derecho Atras (tronco)
    for (int y = 1; y <= 4; y++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(-8, y * 2, -8), 2.0f, wood_material));
    }

    //Pilar Izquierda Atras (tronco)
    for (int y = 1; y <= 4; y++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(2, y * 2, -8), 2.0f, wood_material));
    }
    
    //Base pierda frontal
    for (int x = -3; x <= -1; x++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(x * 2, 2, 2), 2.0f, stone_material));
    }

    //Base piedra trasera
    for (int x = -3; x <= 1; x++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(x * 2, 2, -8), 2.0f, stone_material));
    }

    //Base piedra lateral derecha
    for (int z = -3; z <= 1; z++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(2, 2, z * 2), 2.0f, stone_material));
    }

    //Base piedra lateral izquierda
    for (int z = -3; z <= 1; z++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(-8, 2, z * 2), 2.0f, stone_material));
    }

    // ======================= Pared Ladrillo Frontal =======================
    for (int y = 2; y <= 3; y++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(-6, y * 2, 2), 2.0f, brick_material));
    }

    for (int y = 2; y <= 3; y++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(-2, y * 2, 2), 2.0f, brick_material));
    }

    for (int x = -3; x <= 1; x++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(x * 2, 8, 2), 2.0f, brick_material));
    }

    for (int y = 2; y <= 3; y++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(-4, y * 2, 2), 2.0f, glass_material));
    }

    //Pared Ladrillo trasera
    for (int x = -3; x <= 1; x++) {
        for (int y = 2; y <= 4; y++) {
            raytracer.addObject(std::make_unique<Cube>(Vec3(x * 2, y * 2, -8), 2.0f, brick_material));
        }
    }

    //Pared Ladrillo lateral izquierda
    for (int z = -3; z <= 1; z++) {
        for (int y = 2; y <= 4; y++) {
            raytracer.addObject(std::make_unique<Cube>(Vec3(-8, y * 2, z * 2), 2.0f, brick_material));
        }
    }

    // ======================= Pared Ladrillo lateral derecha =======================
    for (int z = -3; z <= 1; z++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(2, 8, z * 2), 2.0f, brick_material));
    }

    for (int y = 2; y <= 3; y++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(2, y * 2, -6), 2.0f, brick_material));
    }

    for (int y = 2; y <= 3; y++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(2, y * 2, 0), 2.0f, brick_material));
    }

    for (int z = -3; z <= 1; z++) {        // Incrementar desde -2 hasta 0
        for (int y = 2; y <= 3; y++) {
            raytracer.addObject(std::make_unique<Cube>(Vec3(2, y * 2, z * 2), 2.0f, glass_material));
        }
    }

    // Techo (tablones de madera)
    for (int x = -4; x <= 1; x++) {        
        for (int z = -4; z <= 1; z++) {    
            raytracer.addObject(std::make_unique<Cube>(Vec3(x * 2, 10, z * 2), 2.0f, planks_material)); 
        }
    }

    //Tronco de arbol
    for (int y = 1; y <= 5; y++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(10, y * 2, -2), 2.0f, wood_material));
    }

    //Tronco de arbol
    for (int y = 1; y <= 5; y++) {
        raytracer.addObject(std::make_unique<Cube>(Vec3(10, y * 2, -2), 2.0f, wood_material));
    }

    // Copa de hojas 
    for (int x = -1; x <= 1; x++) {        
        for (int z = -1; z <= 1; z++) {     
            for (int y = 0; y <= 2; y++) {  
                raytracer.addObject(std::make_unique<Cube>(
                    Vec3(10 + x * 2, 12 + y * 2, -2 + z * 2), 
                    2.0f, 
                    leaves_material
                ));
            }
        }
    }
    
    std::cout << "Iniciando renderizado del diorama con texturas..." << std::endl;
    raytracer.render();
    std::cout << "¡Renderizado completo! Archivo guardado como output.ppm" << std::endl;
    
    return 0;
}