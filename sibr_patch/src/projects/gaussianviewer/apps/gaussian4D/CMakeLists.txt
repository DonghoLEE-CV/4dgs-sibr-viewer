# SIBR-4D (11-23): YUView-style 4DGS comparison viewer (logic lives in renderer/G4DApp.cpp)
project(SIBR_gaussian4D_app)

file(GLOB SOURCES "*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	sibr_gaussian
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "projects/gaussian/apps")

include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB
	RSC_FOLDER 	"gaussian"
    COMPONENT   ${PROJECT_NAME}_install
)
